# ZepCut

Automatic video editor and local desktop app, written in Python.

ZepCut takes a raw recording and produces a cleaned-up edit:

- cuts silence (FFmpeg `silencedetect`)
- transcribes with word-level timestamps (`faster-whisper`) and removes filler words
- normalises loudness (two-pass)
- adds intro, outro and a watermark
- scores each edit and **routes low-confidence results to a human for review** instead of publishing them
- watch mode for a folder of recordings, live render progress and logging

`app.py` is a small local app (standard-library HTTP server and a JSON API on 127.0.0.1) that opens a single-page UI in an Edge app-mode window. `projects.py` stores project state. `test_zepcut.py` holds automated tests.

## Run

Requirements: Python 3.10+, FFmpeg on PATH, `pip install faster-whisper`.

```
python zepcut.py edit "path\to\recording.mp4" --dry-run
python zepcut.py watch --once
python app.py
```

## Notes

Built with an AI assistant (Claude) and a second model as reviewer. I set the requirements, tested the output against real recordings and decided what to keep. Intro, outro and branding assets are not included.
