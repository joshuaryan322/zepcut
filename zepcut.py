#!/usr/bin/env python3
"""ZepCut - automated video editor core engine (M1-M4).

Pipeline (see SPEC.md): ingest -> analyze -> plan -> score -> render -> report -> route.
M1: silence cuts, two-pass loudness normalization, AMF encode + x264 fallback.
M2: faster-whisper transcription, filler-word removal, confidence routing.
M3: intro/outro stitching and watermark overlay (drop assets into assets/).
M4: watch mode, live render progress, engine logging.

Usage:
    python zepcut.py edit "C:\\path\\to\\recording.mp4" [--dry-run] [--force] [--no-fillers] [--no-branding]
    python zepcut.py redo "C:\\path\\to\\recording.mp4"
    python zepcut.py watch [--once]
    python zepcut.py config
"""

import argparse
import os as _os
WHISPER_THREADS = max(2, (_os.cpu_count() or 4) - 2)

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("zepcut.json")
LOG_PATH = Path(__file__).with_name("zepcut.log")

# under pythonw (no console) stdout/stderr are None and print() would crash
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

DEFAULTS = {
    "videos_dir": str(Path.home() / "Videos"),
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
    # analysis
    "silence_db": -35.0,      # dB threshold below which audio counts as silence
    "min_silence": 1.0,       # seconds of quiet before it counts as a silence
    # transcription
    "language": "en",
    "whisper_model": "small",
    # filler words
    "fillers_simple": ["um", "uh", "uhh", "umm", "er", "ah", "hmm"],
    "fillers_contextual": ["right"],
    "protected_phrases": [
        "right here", "right now", "right there", "right at", "right above",
        "right below", "right through", "right into", "right side",
        "right after", "right before", "all right", "alright",
        "that's right", "you're right", "am i right",
    ],
    "tic_pause": 0.25,        # min gap (s) beside a contextual filler to count as a tic
    "filler_pad": 0.05,       # extra seconds trimmed around a filler word
    "filler_confidence": 0.85,  # min word confidence before a filler may be cut
    "neighbor_confidence": 0.5, # neighbors must decode at least this cleanly
    # cut planning
    "edge_pad": 0.3,          # seconds of quiet kept at video start/end
    "max_pause": 1.5,         # internal pauses longer than this get shortened
    "kept_pause": 0.7,        # what long pauses are shortened to
    "min_segment": 0.4,       # kept pieces shorter than this merge into neighbors
    "min_duration": 25.0,     # inputs shorter than this are skipped as test clips
    # scoring / routing
    "review_threshold": 70,
    # branding (M3) - assets live in <this folder>/assets unless overridden
    "assets_dir": "",
    "watermark_opacity": 0.8,
    "watermark_margin": 24,
    "watermark_position": "bottom-right",  # top-left/top-right/bottom-left/bottom-right
    "intro_min_duration": 90.0,  # recordings shorter than this skip intro/outro
    # loudness
    "target_i": -14.0,        # integrated loudness target (LUFS)
    "target_tp": -1.5,        # true peak ceiling (dBTP)
    "target_lra": 11.0,       # loudness range target
    # encode
    "video_qp": 18,
    "audio_bitrate": "192k",
    # watch mode (M4)
    "watch_poll": 5.0,        # seconds between folder scans
    "watch_stable": 10.0,     # file size must be unchanged this long
}

LOW_NEIGHBOR = 0.6  # neighbor confidence below this marks a filler cut as risky
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".flv"}


class ZepCutSkip(Exception):
    """Non-fatal per-file condition (test clip, existing output, all-silence)."""


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            user = json.load(f)
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            say(f"[config] ignoring unknown keys: {', '.join(sorted(unknown))}")
        cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
    return cfg


def say(msg):
    """Print to console and append to the engine log."""
    print(msg, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def run(cmd, **kwargs):
    """Run a subprocess, capturing text output without raising on failure."""
    return subprocess.run(
        cmd, capture_output=True, text=True, errors="replace", **kwargs
    )


def fmt_ts(seconds):
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m)}:{s:05.2f}"


# ---------------------------------------------------------------- analyze --

def valid_fps(s):
    """Return s if it is a usable fps fraction ('60', '60000/1001'), else None."""
    m = re.fullmatch(r"(\d+)(?:/(\d+))?", s or "")
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2) or 1)
    return s if num > 0 and den > 0 else None


def probe(cfg, path):
    """Return {duration, fps_str, width, height, has_audio} for a media file."""
    r = run([cfg["ffprobe"], "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)])
    if r.returncode != 0:
        raise ZepCutSkip(f"ffprobe could not read the file:\n{r.stderr.strip()}")
    info = json.loads(r.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if video is None:
        raise ZepCutSkip("no video stream found in the file")
    duration = info.get("format", {}).get("duration")
    if duration is None:
        raise ZepCutSkip("the file reports no duration - "
                         "is OBS still writing it? Wait for the recording to finish.")
    # VFR sources report avg_frame_rate '0/0'; fall back rather than emit fps=0/0
    fps_str = (valid_fps(video.get("avg_frame_rate"))
               or valid_fps(video.get("r_frame_rate")) or "60")
    return {
        "duration": float(duration),
        "fps_str": fps_str,
        "width": video.get("width"),
        "height": video.get("height"),
        "has_audio": audio is not None,
    }


def detect_silences(cfg, path, duration):
    """Return [(start, end), ...] silence intervals via ffmpeg silencedetect."""
    filt = f"silencedetect=noise={cfg['silence_db']}dB:d={cfg['min_silence']}"
    r = run([cfg["ffmpeg"], "-hide_banner", "-i", str(path),
             "-af", filt, "-vn", "-f", "null", "-"])
    silences, pending = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            pending = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m:
            end = float(m.group(1))
            if pending is None:
                # some builds print an unparseable start when silence begins
                # at t=0; only assume 0.0 for the very first interval
                pending = 0.0 if not silences else None
            if pending is not None:
                silences.append((pending, end))
            pending = None
    if pending is not None:  # silence ran to end of file
        silences.append((pending, duration))
    return silences


def normalize_word(raw):
    # Whisper emits curly quotes; fold them so protected phrases still match
    raw = (raw.replace("’", "'").replace("‘", "'")
              .replace("“", '"').replace("”", '"'))
    return raw.strip().strip(",.?!;:…\"'-").lower()


def transcribe(cfg, path):
    """Word-level transcript via faster-whisper.

    Returns a list of word dicts, [] when the file contains no speech, or
    None when transcription is unavailable (degrade to silence-only).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    import warnings
    warnings.filterwarnings("ignore")
    try:
        model = WhisperModel(cfg["whisper_model"], device="cpu", compute_type="int8",
                             cpu_threads=WHISPER_THREADS)
        segments, _info = model.transcribe(
            str(path), word_timestamps=True, language=cfg["language"]
        )
        words = []
        for seg in segments:
            seg_words = seg.words or []
            for wi, w in enumerate(seg_words):
                if w.start is None or w.end is None:
                    continue
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "prob": w.probability,
                    "raw": w.word,
                    "norm": normalize_word(w.word),
                    "segment_end": wi == len(seg_words) - 1,
                })
        return words
    except Exception as exc:  # model download failed, corrupt cache, etc.
        say(f"      transcription failed ({exc}) - continuing with silence cuts only")
        return None


# ---------------------------------------------------------------- fillers --

def _protected_phrase(cfg, words, i):
    """Return the protected phrase covering words[i], if any."""
    norm_i = words[i]["norm"]
    for phrase in cfg["protected_phrases"]:
        toks = phrase.lower().split()
        for k, tok in enumerate(toks):
            if tok != norm_i:
                continue
            lo = i - k
            if lo < 0 or lo + len(toks) > len(words):
                continue
            if all(words[lo + j]["norm"] == toks[j] for j in range(len(toks))):
                return phrase
    return None


def detect_fillers(cfg, words, duration):
    """Return filler cut candidates per SPEC.md 5.2 (two tiers with guards)."""
    simple = {f.lower() for f in cfg["fillers_simple"]}
    contextual = {f.lower() for f in cfg["fillers_contextual"]}
    cuts = []
    for i, w in enumerate(words):
        norm = w["norm"]
        if norm not in simple and norm not in contextual:
            continue
        if w["prob"] < cfg["filler_confidence"]:
            continue
        prev = words[i - 1] if i > 0 else None
        nxt = words[i + 1] if i + 1 < len(words) else None
        if norm in contextual:
            gap_prev = (w["start"] - prev["end"]) if prev else 999.0
            gap_next = (nxt["start"] - w["end"]) if nxt else 999.0
            # segment_end alone is unreliable (Whisper breaks segments at
            # ~30s chunk boundaries too) - require terminal punctuation
            seg_end_tic = (w["segment_end"]
                           and w["raw"].rstrip().endswith((".", "?", "!", ",")))
            standalone = (gap_prev >= cfg["tic_pause"]
                          or gap_next >= cfg["tic_pause"]
                          or seg_end_tic)
            if not standalone:
                continue
            if _protected_phrase(cfg, words, i):
                continue
            if prev and prev["prob"] < cfg["neighbor_confidence"]:
                continue
            if nxt and nxt["prob"] < cfg["neighbor_confidence"]:
                continue
        s = w["start"] - cfg["filler_pad"]
        e = w["end"] + cfg["filler_pad"]
        if prev:
            s = max(s, prev["end"])   # never bisect the neighboring word
        if nxt:
            e = min(e, nxt["start"])
        s, e = max(s, 0.0), min(e, duration)
        if e - s < 0.05:
            continue
        cuts.append({
            "start": s, "end": e, "kind": f"filler '{norm}'",
            "contextual": norm in contextual,
            "neighbor_low": bool(
                (prev and prev["prob"] < LOW_NEIGHBOR)
                or (nxt and nxt["prob"] < LOW_NEIGHBOR)
            ),
        })
    return cuts


# ------------------------------------------------------------------- plan --

def _word_between(words, s, e):
    """True if any transcript word's midpoint falls inside (s, e)."""
    return any(s < (w["start"] + w["end"]) / 2 < e for w in (words or []))


def plan_cuts(cfg, silences, filler_cuts, duration, words):
    """Combine silence and filler cuts into kept segments + a final cut list.

    Returns (segments, cuts, applied_fillers). Merging never swallows a real
    word: if the sliver between two cuts contains one, the filler cut is
    dropped instead of merged. Cuts folded into the edge trims stay visible
    in the cut list and the filler accounting.
    """
    start, end = 0.0, duration
    lead_cut, trail_cut = None, None
    internal = []

    for s, e in silences:
        if s <= 0.5 and e >= duration - 0.5:         # the whole file is silent
            return [], [], []
        if s <= 0.5 and e < duration - 0.5:          # lead-in dead air
            new_start = max(0.0, e - cfg["edge_pad"])
            if new_start > start:
                start = new_start
                if lead_cut is None:
                    lead_cut = {"start": 0.0, "end": start,
                                "kind": "lead-in silence"}
                else:
                    lead_cut["end"] = start
        elif e >= duration - 0.5 and s > 0.5:        # trailing dead air
            new_end = min(duration, s + cfg["edge_pad"])
            if new_end < end:
                end = new_end
                if trail_cut is None:
                    trail_cut = {"start": end, "end": duration,
                                 "kind": "trailing silence"}
                else:
                    trail_cut["start"] = end
        elif (e - s) > cfg["max_pause"]:             # long internal pause
            half = cfg["kept_pause"] / 2.0
            internal.append({"start": s + half, "end": e - half,
                             "kind": "long pause tightened", "filler": False})

    if start >= end:
        return [], [], []

    for c in filler_cuts:
        s, e = max(c["start"], start), min(c["end"], end)
        if e - s >= 0.05:
            internal.append({**c, "start": s, "end": e, "filler": True})

    # Pull every cut back off the words before anything else looks at it.
    #
    # silencedetect works on amplitude, so a quiet consonant onset reads as
    # silence: on Episode 2 a cut ran 0.55s into "That's", which would have
    # resumed the episode mid-word. The merge logic below already guards
    # fillers with _word_between, on the stated assumption that "silence cuts
    # never contain words" - this is that assumption failing, so clamp here
    # rather than trust it.
    if words:
        clamped = []
        for c in internal:
            s0, e0 = c["start"], c["end"]
            for w in words:
                if w["end"] > s0 and w["start"] < e0:
                    if w["start"] <= s0:
                        s0 = max(s0, w["end"] + 0.02)
                    else:
                        e0 = min(e0, w["start"] - 0.02)
            if e0 - s0 >= 0.10:
                clamped.append({**c, "start": round(s0, 3), "end": round(e0, 3)})
        internal = clamped

    internal.sort(key=lambda c: c["start"])
    merged = []
    for c in internal:
        if not merged:
            merged.append(dict(c))
            continue
        last = merged[-1]
        gap = c["start"] - last["end"]
        if c["start"] < last["end"] or gap < cfg["min_segment"]:
            if gap > 0 and _word_between(words, last["end"], c["start"]):
                # a real word lives in the sliver - drop the filler cut
                # rather than eat the word (silence cuts never contain words)
                if c.get("filler"):
                    continue
                if last.get("filler"):
                    merged[-1] = dict(c)
                    continue
            if c["kind"] not in last["kind"]:
                last["kind"] = f"{last['kind']} + {c['kind']}"
            last["end"] = max(last["end"], c["end"])
            last["filler"] = last.get("filler") or c.get("filler")
            last["contextual"] = last.get("contextual") or c.get("contextual")
            last["neighbor_low"] = last.get("neighbor_low") or c.get("neighbor_low")
        else:
            merged.append(dict(c))

    # slivers against the trimmed edges fold into the edge trim; the folded
    # cut must stay visible in the report/scoring, so it extends the edge
    # cut entry (or becomes one) instead of vanishing
    folded_fillers = []
    while merged and merged[0]["start"] - start < cfg["min_segment"]:
        c = merged.pop(0)
        if c.get("filler"):
            folded_fillers.append(c)
        if lead_cut is None:
            lead_cut = {"start": start, "end": c["end"], "kind": c["kind"]}
        else:
            lead_cut["end"] = c["end"]
            if c["kind"] not in lead_cut["kind"]:
                lead_cut["kind"] += f" + {c['kind']}"
        start = c["end"]
    while merged and end - merged[-1]["end"] < cfg["min_segment"]:
        c = merged.pop()
        if c.get("filler"):
            folded_fillers.append(c)
        if trail_cut is None:
            trail_cut = {"start": c["start"], "end": end, "kind": c["kind"]}
        else:
            trail_cut["start"] = c["start"]
            if c["kind"] not in trail_cut["kind"]:
                trail_cut["kind"] = f"{c['kind']} + {trail_cut['kind']}"
        end = c["start"]
    kept_total = (end - start) - sum(c["end"] - c["start"] for c in merged)
    if start >= end or kept_total < 1.0:
        return [], [], []

    cuts = [c for c in (lead_cut,) if c]
    segments, cursor = [], start
    for c in merged:
        segments.append((cursor, c["start"]))
        cuts.append(c)
        cursor = c["end"]
    segments.append((cursor, end))
    if trail_cut:
        cuts.append(trail_cut)
    cuts.sort(key=lambda c: c["start"])
    applied_fillers = [c for c in merged if c.get("filler")] + folded_fillers
    return segments, cuts, applied_fillers


def segments_from_cuts(cuts, duration):
    """Complement of a cut list over [0, duration] - used by the review UI
    to re-render with a user-approved subset of cuts."""
    segments, cursor = [], 0.0
    for c in sorted(cuts, key=lambda c: c["start"]):
        s, e = max(0.0, c["start"]), min(duration, c["end"])
        if s > cursor:
            segments.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        segments.append((cursor, duration))
    return [(s, e) for s, e in segments if e - s > 0.01]


# ------------------------------------------------------------------ score --

def score_edit(cfg, duration, kept, cuts, applied_fillers, words):
    """Confidence score per SPEC.md 5.4. Returns (score, notes, hard_flags)."""
    score, notes, hard = 100, [], []
    removed = duration - kept
    minutes = max(duration / 60.0, 1e-9)

    if removed / duration > 0.20:
        score -= 20
        notes.append(f"-20  {100 * removed / duration:.0f}% of runtime removed (>20%)")
    biggest = max((c["end"] - c["start"] for c in cuts), default=0.0)
    if biggest > 8.0:
        score -= 15
        notes.append(f"-15  single cut of {biggest:.1f}s (>8s)")
    if biggest > 20.0:
        hard.append(f"single cut of {biggest:.1f}s exceeds 20s")

    if len(applied_fillers) / minutes > 6:
        score -= 15
        notes.append(f"-15  {len(applied_fillers)} filler cuts "
                     f"({len(applied_fillers) / minutes:.1f}/min, >6/min)")
    ctx = [c for c in applied_fillers if c.get("contextual")]
    if len(ctx) / minutes > 4:
        score -= 15
        notes.append(f"-15  {len(ctx)} contextual-filler cuts "
                     f"({len(ctx) / minutes:.1f}/min, >4/min) - possible misfires")
    risky = sum(1 for c in applied_fillers if c.get("neighbor_low"))
    if risky:
        score -= 5 * risky
        notes.append(f"-{5 * risky}  {risky} filler cut(s) beside low-confidence words")

    if words is None:
        notes.append(" *   transcription unavailable - silence cuts only")
    elif not words:
        hard.append("no speech detected in the recording")
    else:
        mean_p = sum(w["prob"] for w in words) / len(words)
        if mean_p < 0.80:
            score -= 20
            notes.append(f"-20  mean transcript confidence {mean_p:.2f} (<0.80)")

    if len(cuts) / max(duration / 600.0, 1e-9) > 40:
        score -= 10
        notes.append(f"-10  {len(cuts)} cuts (>40 per 10 min)")
    return score, notes, hard


# --------------------------------------------------------------- branding --

def assets(cfg):
    base = Path(cfg["assets_dir"]) if cfg["assets_dir"] else Path(__file__).with_name("assets")
    return {
        "intro": base / "intro.mp4",
        "outro": base / "outro.mp4",
        "watermark": base / "watermark.png",
        "cache": base / ".conformed",
    }


def conform_asset(cfg, src, cache_dir, width, height, fps_str, progress=None):
    """Conform an intro/outro to the target resolution/fps/loudness (cached)."""
    key = f"{src.stem}_{width}x{height}_{fps_str.replace('/', '-')}.mp4"
    out = cache_dir / key
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = probe(cfg, src)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps_str},format=yuv420p")
    af = (f"loudnorm=I={cfg['target_i']}:TP={cfg['target_tp']}:LRA={cfg['target_lra']},"
          f"aresample=48000,aformat=channel_layouts=stereo")
    inputs = ["-i", str(src)]
    maps = ["-map", "0:v:0", "-map", "0:a:0" if info["has_audio"] else "1:a:0"]
    if not info["has_audio"]:  # silent asset still needs an audio track to concat
        inputs += ["-f", "lavfi", "-t", str(info["duration"]),
                   "-i", "anullsrc=r=48000:cl=stereo"]
    tmp = out.with_name(out.stem + ".part.mp4")
    ok = encode(cfg, inputs + maps + ["-vf", vf, "-af", af, "-shortest"],
                tmp, info["duration"], progress)
    if not ok:
        raise ZepCutSkip(f"could not conform branding asset {src.name}")
    tmp.replace(out)
    return out


def concat_branding(cfg, parts, out_path):
    """Concatenate conformed intro/body/outro losslessly (stream copy)."""
    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                      encoding="utf-8")
    for p in parts:
        lst.write("file '" + str(p).replace("'", "'\\''") + "'\n")
    lst.close()
    tmp = out_path.with_name(out_path.stem + ".part.mp4")
    try:
        r = run([cfg["ffmpeg"], "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                 "-i", lst.name, "-c", "copy", str(tmp)])
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            # stream-copy mismatch: fall back to a re-encode concat
            r = run([cfg["ffmpeg"], "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                     "-i", lst.name,
                     "-c:v", "libx264", "-preset", "medium", "-crf", str(cfg["video_qp"]),
                     "-c:a", "aac", "-b:a", cfg["audio_bitrate"], str(tmp)])
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            raise ZepCutSkip(f"branding concat failed:\n{r.stderr[-500:]}")
        tmp.replace(out_path)
    finally:
        os.unlink(lst.name)
        if tmp.exists():
            tmp.unlink()


# ----------------------------------------------------------------- render --

def audio_graph(segments, tail):
    """Build the audio-only trim/concat part of a filter graph + `tail`."""
    parts, labels = [], []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(segments)}:v=0:a=1[ac]")
    parts.append(f"[ac]{tail}[aout]")
    return ";".join(parts)


def graph_script(graph):
    """Write a filter graph to a temp file (avoids the 32k command-line cap
    on Windows when a long recording produces hundreds of segments)."""
    f = tempfile.NamedTemporaryFile("w", suffix=".filtergraph",
                                    delete=False, encoding="utf-8")
    f.write(graph)
    f.close()
    return f.name


def measure_loudness(cfg, path, segments):
    """First loudnorm pass over the planned edit; returns measured stats."""
    tail = (f"loudnorm=I={cfg['target_i']}:TP={cfg['target_tp']}"
            f":LRA={cfg['target_lra']}:print_format=json")
    script = graph_script(audio_graph(segments, tail))
    try:
        r = run([cfg["ffmpeg"], "-hide_banner", "-i", str(path),
                 "-filter_complex_script", script,
                 "-map", "[aout]", "-f", "null", "-"])
    finally:
        os.unlink(script)
    blocks = re.findall(r"\{[^{}]+\}", r.stderr)
    for block in reversed(blocks):
        try:
            stats = json.loads(block)
            if "input_i" in stats:
                return stats
        except json.JSONDecodeError:
            continue
    raise ZepCutSkip(f"loudness measurement failed:\n{r.stderr[-800:]}")


def encode(cfg, ff_args, out_path, total_seconds, progress=None):
    """Run one ffmpeg encode with live progress. Returns True on success."""
    cmd = [cfg["ffmpeg"], "-hide_banner", "-y", "-nostats",
           *ff_args, "-progress", "pipe:1", str(out_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace")
    err_tail = []

    def drain_stderr():
        for line in proc.stderr:
            err_tail.append(line)
            if len(err_tail) > 40:
                err_tail.pop(0)

    t = threading.Thread(target=drain_stderr, daemon=True)
    t.start()
    started = time.time()
    for line in proc.stdout:
        m = re.match(r"out_time_us=(\d+)", line.strip())
        if m and total_seconds > 0:
            done = int(m.group(1)) / 1e6
            pct = min(99.9, 100.0 * done / total_seconds)
            if progress:
                progress(pct)
            else:
                elapsed = time.time() - started
                eta = elapsed / max(pct, 0.1) * (100 - pct)
                print(f"\r      rendering {pct:5.1f}%   ETA {fmt_ts(eta)}   ",
                      end="", flush=True)
    proc.wait()
    t.join(timeout=5)
    if progress is None:
        print("\r" + " " * 48 + "\r", end="", flush=True)
    ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    if not ok:
        encode.last_error = "".join(err_tail)[-800:]
    return ok


encode.last_error = ""


def render(cfg, path, out_path, segments, fps_str, loud,
           watermark=None, progress=None):
    """Render the edited body. Returns the encoder that was used."""
    parts, vlabels, alabels = [], [], []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]")
        vlabels.append(f"[v{i}]")
        alabels.append(f"[a{i}]")
    pairs = "".join(v + a for v, a in zip(vlabels, alabels))
    parts.append(f"{pairs}concat=n={len(segments)}:v=1:a=1[vc][ac]")
    if watermark:
        margin = int(cfg["watermark_margin"])
        pos = {
            "top-left": f"{margin}:{margin}",
            "top-right": f"W-w-{margin}:{margin}",
            "bottom-left": f"{margin}:H-h-{margin}",
            "bottom-right": f"W-w-{margin}:H-h-{margin}",
        }.get(cfg["watermark_position"], f"W-w-{margin}:H-h-{margin}")
        parts.append(f"[vc]fps={fps_str}[vbase]")
        parts.append(f"[1:v]format=rgba,"
                     f"colorchannelmixer=aa={cfg['watermark_opacity']}[wm]")
        parts.append(f"[vbase][wm]overlay={pos}:format=auto[vout]")
    else:
        parts.append(f"[vc]fps={fps_str}[vout]")
    parts.append(
        f"[ac]loudnorm=I={cfg['target_i']}:TP={cfg['target_tp']}:LRA={cfg['target_lra']}"
        f":measured_I={loud['input_i']}:measured_TP={loud['input_tp']}"
        f":measured_LRA={loud['input_lra']}:measured_thresh={loud['input_thresh']}"
        f":offset={loud['target_offset']}:linear=true,"
        f"aresample=48000,aformat=channel_layouts=stereo[aout]"
    )
    graph = ";".join(parts)

    encoders = [
        ("h264_amf", ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp",
                      "-qp_i", str(cfg["video_qp"]), "-qp_p", str(cfg["video_qp"]),
                      "-profile:v", "high"]),
        ("libx264", ["-c:v", "libx264", "-preset", "medium",
                     "-crf", str(cfg["video_qp"])]),
    ]
    total = sum(e - s for s, e in segments)
    inputs = ["-i", str(path)] + (["-i", str(watermark)] if watermark else [])
    # render to a temp name, move into place only on success - a failed or
    # interrupted run must never leave a half-written file in Ready/Review
    tmp_path = out_path.with_name(out_path.stem + ".part.mp4")
    script = graph_script(graph)
    try:
        for name, vargs in encoders:
            ff_args = [*inputs, "-filter_complex_script", script,
                       "-map", "[vout]", "-map", "[aout]", *vargs,
                       "-c:a", "aac", "-b:a", cfg["audio_bitrate"]]
            if encode(cfg, ff_args, tmp_path, total, progress):
                tmp_path.replace(out_path)
                return name
            say(f"[warn] {name} encode failed, trying fallback...")
    finally:
        os.unlink(script)
        if tmp_path.exists():
            tmp_path.unlink()
    raise ZepCutSkip(f"all encoders failed. Last error:\n{encode.last_error}")


# ----------------------------------------------------------------- report --

def build_report(path, out_path, info, segments, cuts, loud, cfg, encoder,
                 words, applied_fillers, score, notes, hard, branding_desc):
    kept = sum(e - s for s, e in segments)
    removed = info["duration"] - kept
    if words:
        mean_p = sum(w["prob"] for w in words) / len(words)
        tline = f"{len(words)} words, mean confidence {mean_p:.2f}"
    elif words is None:
        tline = "unavailable (skipped or faster-whisper missing)"
    else:
        tline = "no speech detected"
    filler_names = [c["kind"] for c in applied_fillers]
    fline = ", ".join(filler_names) if filler_names else "none found"

    lines = [
        f"ZepCut report - {path.name}",
        f"Processed:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Original:   {fmt_ts(info['duration'])}",
        f"Final:      {fmt_ts(kept)}   "
        f"(removed {removed:.1f}s, {100 * removed / info['duration']:.1f}%)",
        f"Loudness:   measured {loud['input_i']} LUFS -> target {cfg['target_i']} LUFS"
        f"   (peak {loud['input_tp']} dBTP -> ceiling {cfg['target_tp']})",
        f"Transcript: {tline}",
        f"Fillers cut: {fline}",
        f"Branding:   {branding_desc}",
        f"Video:      {info['width']}x{info['height']} @ {info['fps_str']} fps, "
        f"{encoder} qp {cfg['video_qp']}",
        f"Audio:      AAC {cfg['audio_bitrate']} 48kHz",
        f"Output:     {out_path}",
        "",
        f"Cuts ({len(cuts)}):",
    ]
    for i, c in enumerate(cuts, 1):
        lines.append(
            f"  {i:2d}. {fmt_ts(c['start'])} - {fmt_ts(c['end'])}"
            f"  {c['kind']:<30s} -{c['end'] - c['start']:.2f}s"
        )
    if not cuts:
        lines.append("  (none - nothing matched the thresholds)")

    lines += ["", f"Confidence: {score}/100"]
    lines += [f"  {n}" for n in notes]
    for h in hard:
        lines.append(f"  FLAG {h}")
    if score >= cfg["review_threshold"] and not hard:
        lines.append(f"Route: Ready (score {score} >= {cfg['review_threshold']})")
    else:
        why = "flagged" if hard else f"score {score} < {cfg['review_threshold']}"
        lines.append(f"Route: Review ({why}) - check the cuts above before uploading")
    return "\n".join(lines)


# ----------------------------------------------------------------- engine --

def process_file(cfg, file, dry_run=False, force=False, no_fillers=False,
                 no_branding=False, overwrite=False, progress=None):
    """Full pipeline for one file. Returns the report text.

    Raises ZepCutSkip for non-fatal conditions. `progress(stage, pct)` is an
    optional callback used by watch mode and the desktop app.
    """
    def stage(name, pct):
        if progress:
            progress(name, pct)

    path = Path(file).resolve()
    if not path.exists():
        raise ZepCutSkip(f"file not found: {path}")

    info = probe(cfg, path)
    if not info["has_audio"]:
        raise ZepCutSkip("file has no audio track; nothing to analyze")
    if info["duration"] < cfg["min_duration"] and not force:
        raise ZepCutSkip(
            f"{fmt_ts(info['duration'])} is under min_duration "
            f"({cfg['min_duration']}s) - looks like a test clip. Use --force to edit anyway."
        )

    videos_dir = Path(cfg["videos_dir"])
    base = videos_dir if path.parent == videos_dir else path.parent
    ready_dir, review_dir = base / "Ready", base / "Review"
    existing = [d / (path.stem + ".mp4") for d in (ready_dir, review_dir)
                if (d / (path.stem + ".mp4")).exists()]
    if existing and not (overwrite or dry_run):
        raise ZepCutSkip(f"output already exists: {existing[0]}\n"
                         f"       use 'zepcut redo' to overwrite it.")

    say(f"[1/5] analyzing silences in {path.name} ...")
    stage("analyzing", 5)
    silences = detect_silences(cfg, path, info["duration"])

    words = None
    if not no_fillers:
        say("[2/5] transcribing (first run downloads the speech model) ...")
        stage("transcribing", 20)
        words = transcribe(cfg, path)
        if words is None:
            say("      transcription unavailable - continuing with silence cuts only")
    else:
        say("[2/5] transcription skipped (--no-fillers)")

    filler_cuts = detect_fillers(cfg, words, info["duration"]) if words else []
    segments, cuts, applied_fillers = plan_cuts(
        cfg, silences, filler_cuts, info["duration"], words
    )
    if not segments:
        raise ZepCutSkip("no speech found - the whole file is silence")

    kept = sum(e - s for s, e in segments)
    score, notes, hard = score_edit(
        cfg, info["duration"], kept, cuts, applied_fillers, words
    )
    out_dir = ready_dir if (score >= cfg["review_threshold"] and not hard) else review_dir
    out_path = out_dir / (path.stem + ".mp4")

    # branding plan (M3)
    a = assets(cfg)
    watermark = a["watermark"] if (not no_branding and a["watermark"].exists()) else None
    use_io = (not no_branding and info["duration"] >= cfg["intro_min_duration"])
    intro = a["intro"] if (use_io and a["intro"].exists()) else None
    outro = a["outro"] if (use_io and a["outro"].exists()) else None
    bits = ([p.name for p in (intro, outro) if p]
            + (["watermark"] if watermark else []))
    branding_desc = " + ".join(bits) if bits else "none (no assets found)"

    say(f"[3/5] measuring loudness across {len(segments)} kept segment(s) ...")
    stage("measuring loudness", 35)
    loud = measure_loudness(cfg, path, segments)

    if dry_run:
        report = build_report(path, out_path, info, segments, cuts, loud, cfg,
                              "(dry run - not rendered)", words, applied_fillers,
                              score, notes, hard, branding_desc)
        say("\n" + report)
        return report

    free = shutil.disk_usage(path.parent).free
    need = 2 * path.stat().st_size
    if free < need:
        raise ZepCutSkip(f"low disk space: {free / 1e9:.1f} GB free, "
                         f"want {need / 1e9:.1f} GB")

    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"[4/5] rendering ({fmt_ts(kept)} final) ...")
    encoder = render_and_brand(cfg, path, out_path, segments, info, loud,
                               watermark, intro, outro,
                               lambda p: stage("rendering", 35 + p * 0.55))

    say("[5/5] writing report ...")
    stage("finishing", 97)
    report = build_report(path, out_path, info, segments, cuts, loud, cfg,
                          encoder, words, applied_fillers, score, notes, hard,
                          branding_desc)
    (out_dir / (path.stem + ".report.txt")).write_text(report, encoding="utf-8")
    # Keep the transcript. It cost five minutes of CPU to produce and the
    # faceless composer needs exactly this data - without it, the same audio
    # gets transcribed a second time.
    if words:
        (out_dir / (path.stem + ".words.json")).write_text(
            json.dumps([{"w": w["norm"], "t": w["start"], "e": w["end"]}
                        for w in words]), encoding="utf-8")
    cut_data = {
        "source": str(path), "score": score, "notes": notes, "flags": hard,
        "original_duration": info["duration"], "final_duration": kept,
        "cuts": [{"start": c["start"], "end": c["end"], "kind": c["kind"]}
                 for c in cuts],
    }
    if out_dir == review_dir:
        (out_dir / (path.stem + ".cuts.json")).write_text(
            json.dumps(cut_data, indent=2), encoding="utf-8"
        )

    # a redo may route differently than the previous run - remove stale copies
    other = review_dir if out_dir == ready_dir else ready_dir
    for stale in (other / (path.stem + ".mp4"), other / (path.stem + ".report.txt"),
                  other / (path.stem + ".cuts.json")):
        if stale.exists():
            stale.unlink()

    stage("done", 100)
    say("\n" + report)
    return report


def render_and_brand(cfg, path, out_path, segments, info, loud,
                     watermark, intro, outro, progress=None):
    """Render the body and stitch branding around it. Returns encoder name."""
    a = assets(cfg)
    body_target = out_path if not (intro or outro) else \
        out_path.with_name(out_path.stem + ".body.mp4")
    encoder = render(cfg, path, body_target, segments, info["fps_str"], loud,
                     watermark=watermark, progress=progress)
    if intro or outro:
        say("      stitching intro/outro ...")
        conformed = [conform_asset(cfg, p, a["cache"], info["width"],
                                   info["height"], info["fps_str"])
                     for p in (intro,) if p]
        conformed.append(body_target)
        conformed += [conform_asset(cfg, p, a["cache"], info["width"],
                                    info["height"], info["fps_str"])
                      for p in (outro,) if p]
        concat_branding(cfg, conformed, out_path)
        body_target.unlink()
    return encoder


def apply_reviewed(cfg, source, keep_cuts, total_cuts, progress=None):
    """Re-render `source` with a user-approved subset of cuts (review UI).

    keep_cuts: the cut dicts the user chose to apply. Output always routes to
    Ready (a human just reviewed it) and Review artifacts are removed.
    """
    def stage(name, pct):
        if progress:
            progress(name, pct)

    path = Path(source).resolve()
    if not path.exists():
        raise ZepCutSkip(f"source not found: {path}")
    info = probe(cfg, path)
    segments = segments_from_cuts(keep_cuts, info["duration"])
    if not segments:
        raise ZepCutSkip("the selected cuts would remove the entire video")

    videos_dir = Path(cfg["videos_dir"])
    base = videos_dir if path.parent == videos_dir else path.parent
    ready_dir, review_dir = base / "Ready", base / "Review"
    ready_dir.mkdir(parents=True, exist_ok=True)
    out_path = ready_dir / (path.stem + ".mp4")

    a = assets(cfg)
    watermark = a["watermark"] if a["watermark"].exists() else None
    use_io = info["duration"] >= cfg["intro_min_duration"]
    intro = a["intro"] if (use_io and a["intro"].exists()) else None
    outro = a["outro"] if (use_io and a["outro"].exists()) else None

    say(f"review-apply: {path.name} with {len(keep_cuts)}/{total_cuts} cuts")
    stage("measuring loudness", 10)
    loud = measure_loudness(cfg, path, segments)
    stage("rendering", 20)
    encoder = render_and_brand(cfg, path, out_path, segments, info, loud,
                               watermark, intro, outro,
                               lambda p: stage("rendering", 20 + p * 0.75))

    kept = sum(e - s for s, e in segments)
    lines = [
        f"ZepCut report - {path.name} (manually reviewed)",
        f"Processed:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Original:   {fmt_ts(info['duration'])}",
        f"Final:      {fmt_ts(kept)}",
        f"Applied {len(keep_cuts)} of {total_cuts} proposed cuts after review:",
    ] + [f"  {fmt_ts(c['start'])} - {fmt_ts(c['end'])}  {c['kind']}"
         for c in sorted(keep_cuts, key=lambda c: c["start"])] + [
        f"Video:      {encoder} qp {cfg['video_qp']}",
        "Route: Ready (human reviewed)",
    ]
    report = "\n".join(lines)
    (ready_dir / (path.stem + ".report.txt")).write_text(report, encoding="utf-8")
    for stale in (review_dir / (path.stem + ".mp4"),
                  review_dir / (path.stem + ".report.txt"),
                  review_dir / (path.stem + ".cuts.json")):
        if stale.exists():
            stale.unlink()
    stage("done", 100)
    say(report)
    return report


# ------------------------------------------------------------------ watch --

def output_stems(cfg):
    base = Path(cfg["videos_dir"])
    stems = set()
    for d in (base / "Ready", base / "Review"):
        if d.exists():
            stems |= {f.stem for f in d.glob("*.mp4")}
    return stems


def cmd_watch(cfg, once=False, stop=None):
    videos = Path(cfg["videos_dir"])
    say(f"watch: monitoring {videos} (Ctrl+C to stop)")
    # NTFS timestamps are unreliable ("tunneling" can reuse old creation
    # times), so "new" = a filename that was not present when watch started.
    # --once instead processes the existing backlog of unprocessed files.
    initial = set() if once else {f.name for f in videos.iterdir() if f.is_file()}
    sizes = {}       # path -> (size, first_seen_stable)
    while True:
        done_stems = output_stems(cfg)
        candidates = [f for f in videos.iterdir()
                      if f.suffix.lower() in VIDEO_EXTS and f.is_file()
                      and f.stem not in done_stems and f.name not in initial]
        for f in candidates:
            size = f.stat().st_size
            prev = sizes.get(f)
            if prev and prev[0] == size:
                if time.time() - prev[1] >= cfg["watch_stable"]:
                    say(f"watch: processing {f.name}")
                    try:
                        process_file(cfg, f)
                    except ZepCutSkip as exc:
                        say(f"watch: skipped {f.name}: {exc}")
                    except Exception as exc:
                        say(f"watch: ERROR on {f.name}: {exc}")
                    sizes.pop(f, None)
            else:
                sizes[f] = (size, time.time())
        if once and not candidates:
            say("watch: backlog clear, exiting (--once)")
            return
        waited = 0.0
        while waited < cfg["watch_poll"]:      # nap in slices so a stop
            if stop is not None and stop.is_set():   # request lands quickly
                say("watch: stopped")
                return
            time.sleep(0.5)
            waited += 0.5


# -------------------------------------------------------------------- cli --

def cmd_config(cfg):
    print(f"config file: {CONFIG_PATH} "
          f"({'exists' if CONFIG_PATH.exists() else 'missing - using defaults'})")
    for k, v in cfg.items():
        print(f"  {k:<20s} = {v}")
    for tool in ("ffmpeg", "ffprobe"):
        r = run([cfg[tool], "-version"])
        ok = r.stdout.splitlines()[0] if r.returncode == 0 and r.stdout else "NOT FOUND"
        print(f"  [{tool}] {ok}")
    try:
        import faster_whisper
        print(f"  [faster-whisper] {faster_whisper.__version__}")
    except ImportError:
        print("  [faster-whisper] NOT INSTALLED - filler removal disabled")
    a = assets(cfg)
    for k in ("intro", "outro", "watermark"):
        print(f"  [asset:{k}] {'found' if a[k].exists() else 'not present'} "
              f"({a[k]})")


def main():
    # keep unicode filenames/transcripts from crashing redirected cp1252 output
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(prog="zepcut", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("edit", "redo"):
        p = sub.add_parser(name)
        p.add_argument("file")
        p.add_argument("--dry-run", action="store_true",
                       help="analyze and print the report without rendering")
        p.add_argument("--force", action="store_true",
                       help="edit files shorter than min_duration")
        p.add_argument("--no-fillers", action="store_true",
                       help="skip transcription and filler-word removal")
        p.add_argument("--no-branding", action="store_true",
                       help="skip intro/outro and watermark even if assets exist")
    w = sub.add_parser("watch")
    w.add_argument("--once", action="store_true",
                   help="process the current backlog, then exit")
    sub.add_parser("config")
    args = ap.parse_args()

    cfg = load_config()
    if args.cmd == "config":
        cmd_config(cfg)
    elif args.cmd == "watch":
        try:
            cmd_watch(cfg, once=args.once)
        except KeyboardInterrupt:
            say("watch: stopped")
    else:
        try:
            process_file(cfg, args.file, dry_run=args.dry_run, force=args.force,
                         no_fillers=args.no_fillers, no_branding=args.no_branding,
                         overwrite=(args.cmd == "redo"))
        except ZepCutSkip as exc:
            say(f"[skip] {exc}")
            sys.exit(2)


if __name__ == "__main__":
    main()
