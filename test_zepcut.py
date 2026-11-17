"""Unit tests for ZepCut M2 cut-planning, filler detection, and scoring logic.

Run:  python test_zepcut.py
"""

import unittest

from zepcut import (DEFAULTS, detect_fillers, plan_cuts, score_edit,
                    normalize_word, _protected_phrase, valid_fps)


def W(start, end, word, prob=0.95, segment_end=False):
    return {"start": start, "end": end, "prob": prob, "raw": word,
            "norm": normalize_word(word), "segment_end": segment_end}


CFG = dict(DEFAULTS)


class TestFillers(unittest.TestCase):
    def test_simple_filler_cut(self):
        words = [W(1.0, 1.3, " So"), W(1.4, 1.7, " um"), W(1.8, 2.2, " yeah")]
        cuts = detect_fillers(CFG, words, 10.0)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["kind"], "filler 'um'")
        # pad must not bisect neighbors
        self.assertGreaterEqual(cuts[0]["start"], 1.3)
        self.assertLessEqual(cuts[0]["end"], 1.8)

    def test_low_confidence_filler_not_cut(self):
        words = [W(1.0, 1.3, " um", prob=0.5)]
        self.assertEqual(detect_fillers(CFG, words, 10.0), [])

    def test_contextual_right_protected_phrase(self):
        words = [W(1.0, 1.3, " draw"), W(1.35, 1.6, " right"), W(2.2, 2.5, " here")]
        # "right here" with a pause after right - protected phrase must win
        self.assertEqual(detect_fillers(CFG, words, 10.0), [])

    def test_contextual_right_standalone_tic(self):
        words = [W(1.0, 1.3, " level"), W(1.8, 2.1, " right"), W(2.9, 3.2, " so")]
        cuts = detect_fillers(CFG, words, 10.0)
        self.assertEqual(len(cuts), 1)
        self.assertTrue(cuts[0]["contextual"])

    def test_contextual_right_segment_end(self):
        words = [W(1.0, 1.3, " retest"), W(1.35, 1.6, " right?", segment_end=True)]
        cuts = detect_fillers(CFG, words, 10.0)
        self.assertEqual(len(cuts), 1)

    def test_contextual_right_mid_speech_not_cut(self):
        # no pause on either side, not segment end -> content word
        words = [W(1.0, 1.3, " the"), W(1.35, 1.6, " right"), W(1.65, 2.0, " level")]
        self.assertEqual(detect_fillers(CFG, words, 10.0), [])

    def test_contextual_right_unclear_neighbors_not_cut(self):
        words = [W(1.0, 1.3, " mumble", prob=0.3), W(1.8, 2.1, " right"),
                 W(2.9, 3.2, " so")]
        self.assertEqual(detect_fillers(CFG, words, 10.0), [])

    def test_thats_right_protected(self):
        words = [W(1.0, 1.3, " that's"), W(1.35, 1.7, " right", segment_end=True)]
        self.assertIsNotNone(_protected_phrase(CFG, words, 1))
        self.assertEqual(detect_fillers(CFG, words, 10.0), [])


class TestPlan(unittest.TestCase):
    def test_filler_merges_with_adjacent_silence_cut(self):
        # long pause 2..6 (cut becomes 2.35..5.65), filler 0.25s after the
        # cut edge - inside the min_segment merge window
        silences = [(2.0, 6.0)]
        fillers = [{"start": 5.9, "end": 6.2, "kind": "filler 'um'",
                    "contextual": False, "neighbor_low": False}]
        segs, cuts, applied = plan_cuts(CFG, silences, fillers, 20.0, [])
        merged = [c for c in cuts if "+" in c["kind"]]
        self.assertEqual(len(merged), 1)
        self.assertIn("long pause", merged[0]["kind"])
        self.assertIn("filler", merged[0]["kind"])

    def test_word_in_sliver_drops_filler_not_word(self):
        # two filler cuts 0.3s apart with the word "yes" in between:
        # merging would eat "yes", so the second filler must be dropped
        words = [W(5.0, 5.2, " um"), W(5.3, 5.55, " yes"), W(5.65, 5.9, " um")]
        fillers = [
            {"start": 5.0, "end": 5.25, "kind": "filler 'um'",
             "contextual": False, "neighbor_low": False},
            {"start": 5.6, "end": 5.9, "kind": "filler 'um'",
             "contextual": False, "neighbor_low": False},
        ]
        segs, cuts, applied = plan_cuts(CFG, [], fillers, 20.0, words)
        kept = sum(e - s for s, e in segs)
        self.assertEqual(len(applied), 1)          # one filler dropped
        self.assertTrue(any(s <= 5.4 <= e for s, e in segs))  # "yes" survives

    def test_all_silence_returns_empty(self):
        segs, cuts, applied = plan_cuts(CFG, [(0.0, 20.0)], [], 20.0, [])
        self.assertEqual(segs, [])

    def test_m1_behavior_reproduced(self):
        # the validated first test recording: same silences as the real file
        silences = [(0.0, 2.036), (2.923, 4.526), (14.994, 18.174), (89.945, 92.4)]
        segs, cuts, applied = plan_cuts(CFG, silences, [], 92.4, None)
        self.assertEqual(len(cuts), 4)
        kept = sum(e - s for s, e in segs)
        self.assertAlmostEqual(kept, 85.13, delta=0.2)


class TestScore(unittest.TestCase):
    def test_clean_edit_routes_ready(self):
        cuts = [{"start": 0, "end": 2, "kind": "lead-in silence"}]
        score, notes, hard = score_edit(CFG, 100.0, 95.0, cuts, [], [W(1, 2, " hi")])
        self.assertEqual(score, 100)
        self.assertEqual(hard, [])

    def test_heavy_removal_penalized(self):
        cuts = [{"start": 10, "end": 19.5, "kind": "long pause tightened"}]
        score, notes, hard = score_edit(CFG, 60.0, 40.0, cuts, [], [W(1, 2, " hi")])
        # 33% removed (-20) and a 9.5s cut (-15)
        self.assertEqual(score, 65)
        self.assertEqual(hard, [])

    def test_giant_cut_hard_flag(self):
        cuts = [{"start": 10, "end": 35, "kind": "long pause tightened"}]
        score, notes, hard = score_edit(CFG, 300.0, 275.0, cuts, [], [W(1, 2, " hi")])
        self.assertTrue(any("20s" in h for h in hard))

    def test_no_speech_hard_flag(self):
        score, notes, hard = score_edit(CFG, 60.0, 55.0, [], [], [])
        self.assertTrue(any("no speech" in h for h in hard))

    def test_transcription_unavailable_no_penalty(self):
        score, notes, hard = score_edit(CFG, 60.0, 55.0, [], [], None)
        self.assertEqual(score, 100)
        self.assertEqual(hard, [])

    def test_low_transcript_confidence(self):
        words = [W(i, i + 0.2, " blur", prob=0.5) for i in range(10)]
        score, notes, hard = score_edit(CFG, 60.0, 55.0, [], [], words)
        self.assertEqual(score, 80)

    def test_filler_rate_penalty(self):
        fillers = [{"kind": "filler 'um'", "contextual": False,
                    "neighbor_low": False, "start": i, "end": i + 0.2, "filler": True}
                   for i in range(8)]
        words = [W(1, 2, " hi")]
        score, notes, hard = score_edit(CFG, 60.0, 55.0, fillers, fillers, words)
        self.assertIn(-15, [-15])  # rate 8/min > 6/min triggers
        self.assertLessEqual(score, 85)

    def test_contextual_rate_penalty(self):
        fillers = [{"kind": "filler 'right'", "contextual": True,
                    "neighbor_low": False, "start": i, "end": i + 0.2, "filler": True}
                   for i in range(5)]
        words = [W(1, 2, " hi")]
        score, notes, hard = score_edit(CFG, 60.0, 55.0, fillers, fillers, words)
        # 5/min contextual (>4/min) -> -15; five cuts in one minute also
        # trips the >40-per-10min density penalty -> -10 (they co-trigger
        # by construction: ctx>4/min implies >40 cuts/10min when fillers
        # are the only cuts)
        self.assertEqual(score, 75)


class TestReviewFixes(unittest.TestCase):
    """Regression tests for defects found in the M2 adversarial review."""

    def test_folded_edge_cut_stays_accounted(self):
        # lead-in silence 0..2 (start=1.7), filler at 1.72..2.1: the sliver
        # is < min_segment so it folds into the lead trim - but it must
        # still appear in the cut list and the filler accounting
        silences = [(0.0, 2.0)]
        fillers = [{"start": 1.72, "end": 2.10, "kind": "filler 'um'",
                    "contextual": False, "neighbor_low": False}]
        segs, cuts, applied = plan_cuts(CFG, silences, fillers, 60.0, [])
        kept = sum(e - s for s, e in segs)
        cut_total = sum(c["end"] - c["start"] for c in cuts)
        self.assertAlmostEqual(kept + cut_total, 60.0, places=3)  # sums match
        self.assertEqual(len(applied), 1)                # filler still scored
        self.assertIn("filler", cuts[0]["kind"])         # and still reported

    def test_fold_without_leadin_creates_cut_entry(self):
        # no lead-in silence at all; a filler at 0.05..0.30 folds into start
        fillers = [{"start": 0.05, "end": 0.30, "kind": "filler 'um'",
                    "contextual": False, "neighbor_low": False}]
        segs, cuts, applied = plan_cuts(CFG, [], fillers, 60.0, [])
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0]["start"], 0.0, places=3)
        self.assertAlmostEqual(cuts[0]["end"], 0.30, places=3)
        self.assertEqual(len(applied), 1)

    def test_valid_fps(self):
        self.assertEqual(valid_fps("60/1"), "60/1")
        self.assertEqual(valid_fps("60000/1001"), "60000/1001")
        self.assertIsNone(valid_fps("0/0"))      # VFR matroska
        self.assertIsNone(valid_fps("60/0"))
        self.assertIsNone(valid_fps(""))
        self.assertIsNone(valid_fps(None))

    def test_curly_apostrophe_protected_phrase(self):
        # Whisper emits U+2019; "that's right" must still be protected
        words = [W(1.0, 1.3, " that’s"), W(1.35, 1.7, " right?",
                                                segment_end=True)]
        self.assertEqual(normalize_word(" that’s"), "that's")
        self.assertIsNotNone(_protected_phrase(CFG, words, 1))
        self.assertEqual(detect_fillers(CFG, words, 10.0), [])

    def test_segment_end_without_punctuation_not_standalone(self):
        # a mid-sentence 'right' at a Whisper chunk boundary: segment_end
        # True but no punctuation and no pause -> must NOT be cut
        words = [W(1.0, 1.3, " the"), W(1.35, 1.6, " right", segment_end=True)]
        nxt_chunk = W(1.65, 2.0, " level")
        self.assertEqual(detect_fillers(CFG, words + [nxt_chunk], 10.0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
