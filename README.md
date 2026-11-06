# Review Recorder

Record a video review (you talking over a video while it plays) **as smooth as the video really plays**, with **no time
limit**, then turn it into notes an AI like **Claude** can act on: every comment with its time, a picture of the
screen at that moment, and **the exact second of the reviewed video** it refers to.

Built because screen-recorder reviews looked choppy (a 30 fps recording of a 30 fps video drifts and shows fake
judder), had time limits, and the AI reading them had to guess which shot each note meant.

## What you get

```
<review folder>/
  review.mp4        the recording (Windows recorder: track 1 = mic + PC sound, 2 = mic raw, 3 = PC sound)
  words.txt         what was said, with times
  sheets/NN.jpg     the screen at the start of each comment (3 x 3 per page)
  overview/NN.jpg   the screen every 10 s
  REVIEW.md         one page: smoothness check + table  rec time | video time | what was said | sheet
```

Example `REVIEW.md` row: `| 1:49.7 | 0:38.7 | She should be looking at the woman here. | 02 |`
(the comment at 1:49.7 of the recording is about second 0:38.7 of the reviewed video).
Give `REVIEW.md` + the sheets to Claude and ask it to turn them into a fix list per shot.

## Windows: record + read

The recorder grabs the screen on the GPU (ffmpeg `ddagrab`) and keeps **every real screen change with its true time**
(up to the screen's refresh rate), so a 30 or 60 fps video is captured frame for frame. In tests with
frame-numbered clips playing full screen: **0 of 300 frames missed (30 fps), 0 of 600 (60 fps)**; a fixed 60 fps
capture missed 5-13 %. Mic and PC sound are separate tracks; the mic is kept raw (no filters). Live mic meter.

1. Install [Python 3.10+](https://www.python.org/) and an ffmpeg build with `ddagrab` (e.g. the "full" build from
   gyan.dev, ffmpeg 6.1 or newer) on your PATH.
2. In this folder: `python -m venv .venv` then `.venv\Scripts\pip install -r requirements-windows.txt`
3. Start: double-click `Review Recorder.bat` (or `.venv\Scripts\python review_recorder.py`).
4. Type what you are reviewing, pick your mic (a headset beats the laptop mic), press **Start review**, play the
   video, talk. Press **Stop** (window or tray icon). The reader runs by itself after Stop.
5. To add the reviewed video's seconds: `python read_review.py "<review folder>" --render "<the video>.mp4"`

Settings (optional) go in `review_settings.json` next to the script: `out_dir` (default `~/Videos/Reviews`),
`encoder` (`amf` / `nvenc` / `qsv` / `x264`, picked automatically on first use), `sync_fix` (seconds).

## Mac: record with macOS, read with this tool

The Windows recorder does not run on a Mac. Use the Mac's own recorder instead; nothing extra is installed for it.

1. Press **Cmd + Shift + 5** → *Record Entire Screen* → **Options** → *Microphone*: your headset or built-in mic.
   (The Mac recorder can't capture the video's own sound without extra software; the reader only needs your voice
   and the picture.)
2. Play the video and talk. Stop with the stop button in the menu bar. You get a `.mov` file.
3. One-time setup (Terminal): install [Homebrew](https://brew.sh), then `brew install ffmpeg python`, then in this
   folder `python3 -m venv .venv && .venv/bin/pip install -r requirements-reader.txt`
4. Read it: `.venv/bin/python read_review.py "Screen Recording ....mov" --render "<the video>.mp4"`
   This makes a `<recording name> review/` folder next to the recording.

## What it does and doesn't touch (safe to run)

- **Reads** the recording and the reviewed video; **writes only** into one new review folder.
- Deletes only its own temporary files (`_audio_for_whisper.wav`, and its own `NN.jpg` sheets when you re-run it).
- No admin rights, no system settings, no background services. Everything stays inside the `.venv` folder;
  delete the folder to remove it completely.
- **Network:** none, except the first run downloads the Whisper speech model once (`medium` about 1.5 GB on CPU /
  Mac, `large-v3` about 3 GB on an NVIDIA GPU) from Hugging Face into your user cache. Recordings never leave
  your computer.
- Transcribing on a Mac CPU takes a while (roughly real time with `medium`); use `--model small` for a faster,
  rougher transcript.

## Reader options

```
python read_review.py <review folder | recording file> [--render <video>] [--lang en] [--model medium]
```

- `--lang`: language of the speech (default: auto-detect).
- `--model`: Whisper model (default: `large-v3` on an NVIDIA GPU, `medium` on CPU).
- Re-running with `--render` reuses the saved transcript (`segments.json`), so it's quick.
- Playback check: on a Windows recording it lists hitches (pauses longer than ~85 ms while the video plays).
  Constant-frame-rate files (most Mac recordings) repeat frames, so the check can't see hitches there.

## Tests

`tests/make_test_clip.py` makes frame-numbered 30 / 60 fps clips with a flash + beep; `tests/measure_capture.py`
counts missed frames and the sound / picture offset in a recording of them.

## Status

Built and tested on Windows 11 (AMD + NVIDIA laptop) in November 2026. The reader is plain Python + ffmpeg and is
written for macOS too, but has not been run on a Mac yet. Issues and fixes welcome.

## License

MIT - see [LICENSE](LICENSE). By Carl Owen Belen.
