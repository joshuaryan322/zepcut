"""
Series projects for ZepCut.

The graphics stage was originally hardcoded to one series. This makes it a
first-class, switchable thing so the same install can drive several series.

A project is just four folders:
    scripts   - the episode scripts the cues are read from
    episodes  - EPnn-*/graphics/ holding that episode's rendered assets
    autoedit  - where plan-epNN.json files live
    masters   - alpha .mov masters (overlays use these, cutaways use the mp4s)

Nothing here renders. It only answers "what does this project have, and is the
graphics stage switched on?" - the UI and the pipeline both read from it.
"""
import json, re, os
from pathlib import Path

PATH = Path(__file__).with_name("projects.json")


def load():
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"active": None, "graphics_enabled": False, "projects": {}}


def save(d):
    PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")


def active(d=None):
    d = d or load()
    key = d.get("active")
    p = d.get("projects", {}).get(key)
    return (key, p) if p else (None, None)


def episode_of(filename, d=None):
    """Work out which episode a recording belongs to, from its filename."""
    _, p = active(d)
    pat = (p or {}).get("episode_pattern", r"EP(\d{1,2})")
    m = re.search(pat, os.path.basename(filename), re.I)
    return int(m.group(1)) if m else None


def episodes(d=None):
    """Summarise every episode in the active project: assets + plan coverage."""
    key, p = active(d)
    if not p:
        return []
    epdir, autoedit = Path(p["episodes"]), Path(p["autoedit"])
    out = []
    if not epdir.exists():
        return out
    for folder in sorted(epdir.iterdir()):
        if not folder.is_dir():
            continue
        m = re.match(r"EP(\d{2})-(.+)", folder.name)
        if not m:
            continue
        n = int(m.group(1))
        gdir = folder / "graphics"
        assets = sorted(f.name for f in gdir.glob("*.mp4")) if gdir.exists() else []
        script = next((f.name for f in folder.glob("SCRIPT-*")), None)
        plan_file = autoedit / f"plan-ep{n:02d}.json"
        placed, unmatched, events = 0, 0, []
        if plan_file.exists():
            try:
                pl = json.loads(plan_file.read_text(encoding="utf-8"))
                events = pl.get("events", [])
                placed = len(events)
                unmatched = len(pl.get("unmatched", []))
            except Exception:
                pass
        used = {e["asset"] for e in events}
        out.append({
            "n": n,
            "title": m.group(2).replace("-", " "),
            "folder": folder.name,
            "script": script,
            "assets": len(assets),
            "placed": placed,
            "unused": [a for a in assets if a not in used],
            "unmatched": unmatched,
            "has_plan": plan_file.exists(),
            "events": [{"asset": e["asset"], "placement": e["placement"],
                        "line": e["line"], "anchor": e["anchor"][:90],
                        "cue": e["cue"][:110]} for e in events],
        })
    return out
