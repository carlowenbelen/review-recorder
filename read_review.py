"""Read a video review for an AI (e.g. Claude): words with times, frame sheets, a smoothness check, and the matching
second of the reviewed video for every comment. Works on Windows and macOS.

    python read_review.py "<review folder from review_recorder.py>"  [--render "<the video you reviewed>"]
    python read_review.py "<any screen recording, e.g. a Mac .mov>"   [--render "<the video you reviewed>"]
    options: --lang en (default: auto-detect) --model large-v3 (default: large-v3 on an NVIDIA GPU, medium on CPU)

Writes (into the review folder, or "<recording name> review/" next to a single recording):
  words.txt        what was said, with times ([m:ss.s-m:ss.s] text) - faster-whisper
  segments.json    the same as data (reused on re-runs, so --render can be added later without a new transcription)
  sheets/NN.jpg    the screen at the start of each comment (3x3, labelled with the recording time + words)
  overview/NN.jpg  the screen every 10 s (3x3)
  REVIEW.md        one page: playback smoothness check, then  rec time | video time | what was said | sheet
With --render, each comment gets the matching second of that video: the player area is found from the pixels that
change most often, then small grey frames are matched (works for 9:16 videos inside a wide player too).
The mic recording is never changed: Whisper hears a temporary levelled copy that is deleted after.
Optional: env HEAVY_JOB_DIR = a folder with heavy_job.py (a one-job-at-a-time GPU lock) to queue the transcription.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
if os.environ.get("HEAVY_JOB_DIR"):
    sys.path.insert(0, os.environ["HEAVY_JOB_DIR"])


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW, **kw)


def duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], text=True)
    return float(r.stdout.strip() or 0)


def dims(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(path)], text=True)
    w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
    return w, h


# ---------------------------------------------------------------- words
def load_whisper(model_name):
    from faster_whisper import WhisperModel
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return WhisperModel(model_name or "large-v3", device="cuda", compute_type="float16"), "cuda"
    except Exception:  # noqa: BLE001
        pass
    return WhisperModel(model_name or "medium", device="cpu", compute_type="int8"), "cpu"


def transcribe(audio_src, workdir, lang, model_name, label):
    """audio_src = the mic recording (or a screen recording with the mic in its first audio track)."""
    even = workdir / "_audio_for_whisper.wav"
    r = run(["ffmpeg", "-v", "error", "-y", "-i", str(audio_src), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
             "-af", "dynaudnorm=f=250:g=15:m=20", str(even)])
    if r.returncode or not even.exists():
        print("no audio track found - no transcript", flush=True)
        return []
    try:
        from heavy_job import heavy
        lock = heavy("REVIEW READ: " + label)
    except ImportError:
        lock = contextlib.nullcontext()
    try:
        with lock:
            model, dev = load_whisper(model_name)
            segs, _ = model.transcribe(str(even), language=lang, word_timestamps=True, vad_filter=True, beam_size=5)
            out = [{"s": round(s.start, 2), "e": round(s.end, 2), "t": s.text.strip()} for s in segs if s.text.strip()]
    finally:
        even.unlink(missing_ok=True)
    print(f"whisper on {dev}: {len(out)} lines", flush=True)
    return out


# ---------------------------------------------------------------- frames
def grab(video, t, w=640, h=360):
    r = run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.2f}", "-i", str(video), "-frames:v", "1", "-vf",
             f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
             "-f", "image2pipe", "-vcodec", "png", "-"])
    try:
        return Image.open(io.BytesIO(r.stdout)).convert("RGB")
    except Exception:  # noqa: BLE001
        return Image.new("RGB", (w, h))


def label_font(size=18):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheets(video, items, outdir, cols=3, rows=3):
    """items = [(time, label)]; returns the sheet number of each item."""
    outdir.mkdir(exist_ok=True)
    for old in outdir.glob("[0-9][0-9].jpg"):          # only the sheets this script made before (01.jpg, 02.jpg ...)
        old.unlink()
    font = label_font()
    per = cols * rows
    where = []
    for s in range(0, len(items), per):
        page = Image.new("RGB", (640 * cols, 360 * rows), (20, 20, 20))
        for k, (t, label) in enumerate(items[s:s + per]):
            im = grab(video, t)
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 640, 30), fill=(0, 0, 0))
            d.text((8, 5), f"R {mmss(t)}  {label[:52]}", fill=(255, 230, 0), font=font)
            page.paste(im, ((k % cols) * 640, (k // cols) * 360))
            where.append(s // per + 1)
        page.save(outdir / f"{s // per + 1:02d}.jpg", quality=82)
    return where


# ---------------------------------------------------------------- smoothness
def playback_check(video):
    """A variable-frame-rate recording keeps only real screen changes, each with its true time. While a video plays,
    the screen changes at the video's frame rate; a pause longer than ~2.5 frames inside a playing stretch is a hitch.
    (A constant-frame-rate recording, e.g. most Mac / Loom files, can't show hitches this way: the check says so.)"""
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
             str(video)], text=True)
    ts = np.array(sorted(float(x.strip(",")) for x in r.stdout.split() if x.strip(",") not in ("", "N/A")))
    if len(ts) < 10:
        return "too few frames to judge", []
    secs = np.floor(ts).astype(int)
    counts = np.bincount(secs)
    playing = counts >= 20
    hitches = []
    d = np.diff(ts)
    for i, gap in enumerate(d):
        if gap > 0.085 and playing[secs[i]] and playing[secs[i + 1]]:
            hitches.append((ts[i], gap))
    n_play = int(playing.sum())
    rate = float(np.median(counts[playing])) if n_play else 0
    cfr = len(counts) > 5 and np.std(counts[1:-1]) < 1.0
    summary = (f"{len(ts)} screen updates; video playing in {n_play} s of {len(counts)} s; "
               f"typical {rate:.0f} updates per second while playing; {len(hitches)} hitch(es) > 85 ms")
    if cfr:
        summary += " (constant-frame-rate file: repeated frames hide hitches, so this check can't see them)"
    return summary, hitches


# ---------------------------------------------------------------- match to the render
def small(src, fps, vf="", W=64, H=36):
    r = run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"fps={fps},{vf}scale={W}:{H},format=gray", "-f", "rawvideo", "-"])
    a = np.frombuffer(r.stdout, np.uint8)
    return a[: len(a) // (W * H) * W * H].reshape(-1, H * W).astype(np.float32)


def norm(a):
    a = a - a.mean(1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)


def player_area(video, aspect=None):
    """Where the video plays on screen (full screen or a player window): the pixels that change OFTEN (in most seconds
    while it plays), not the ones that changed a lot once (opening the file, switching windows, the mouse).
    With aspect (the render's width / height) the box is snapped to the render's shape (players centre the picture)."""
    sw, sh = dims(video)
    r = run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=1,scale=192:108,format=gray", "-f", "rawvideo", "-"])
    a = np.frombuffer(r.stdout, np.uint8)
    a = a[: len(a) // (192 * 108) * 192 * 108].reshape(-1, 108, 192).astype(np.float32)
    if len(a) < 5:
        return None
    often = (np.abs(np.diff(a, axis=0)) > 6).mean(0) > 0.25     # changed in > 25 % of the seconds

    def run_of(p):                                  # longest stretch of columns / rows that change often
        if p.max() <= 0:
            return 0, -1
        on, best, i = p > 0.45 * p.max(), (0, -1), 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                best = max(best, (i, j), key=lambda b: b[1] - b[0])
                i = j + 1
            else:
                i += 1
        return best
    (x0, x1), (y0, y1) = run_of(often.mean(0)), run_of(often.mean(1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    fx, fy = sw / 192, sh / 108
    bx, by, bw, bh = x0 * fx, y0 * fy, (x1 - x0 + 1) * fx, (y1 - y0 + 1) * fy
    if aspect and bw / bh > aspect * 1.05:          # too wide: keep the height, centre the render's width
        nw = bh * aspect
        bx, bw = bx + (bw - nw) / 2, nw
    elif aspect and bw / bh < aspect / 1.05:        # too tall: keep the width, centre the render's height
        nh = bw / aspect
        by, bh = by + (bh - nh) / 2, nh
    bx, by = max(0, int(bx)), max(0, int(by))
    return bx, by, min(int(bw), sw - bx) // 2 * 2, min(int(bh), sh - by) // 2 * 2


def match(video, render):
    rw, rh = dims(render)
    W, H = (36, 64) if rh > rw else (64, 36)       # compare at the render's own shape
    ref = norm(small(render, 4, W=W, H=H))
    box = player_area(video, rw / rh)
    full = norm(small(video, 2))
    if box:
        win = norm(small(video, 2, f"crop={box[2]}:{box[3]}:{box[0]}:{box[1]},", W=W, H=H))
        full = win if (W, H) != (64, 36) else full  # a squashed full screen can't match a 9:16 render
    else:
        win = full
    out = []
    for i in range(min(len(full), len(win))):
        best = max(((float((ref @ L[i]).max()), int((ref @ L[i]).argmax()) / 4) for L in (full, win)))
        out.append((i / 2, best[1], best[0]))
    return out, box


def at(mapping, t):
    if not mapping:
        return None
    k = min(len(mapping) - 1, int(round(t * 2)))
    for j in (k, k + 1, k + 2, k - 1):        # first good match at or just after the moment the comment starts
        if 0 <= j < len(mapping) and mapping[j][2] > 0.8:
            return mapping[j][1] + (t - mapping[j][0]) if j == k else mapping[j][1]
    return None


# ---------------------------------------------------------------- main
def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    lang, model_name = arg("--lang"), arg("--model")
    render = Path(arg("--render")) if arg("--render") else None
    if src.is_file():                                   # a single screen recording (Mac, Loom download, ...)
        video, folder = src, src.parent / f"{src.stem} review"
        folder.mkdir(exist_ok=True)
        info, audio = {"name": src.stem}, src
    else:                                               # a review_recorder.py folder
        folder, video = src, src / "review.mp4"
        info = json.loads((folder / "info.json").read_text(encoding="utf-8")) if (folder / "info.json").exists() else {}
        audio = folder / "my mic (raw).flac" if (folder / "my mic (raw).flac").exists() else video
    length = duration(video)

    seg_file = folder / "segments.json"
    segs = json.loads(seg_file.read_text(encoding="utf-8")) if seg_file.exists() else None
    if segs is None:
        segs = transcribe(audio, folder, lang, model_name, folder.name)
        seg_file.write_text(json.dumps(segs, ensure_ascii=False, indent=0), encoding="utf-8")
        (folder / "words.txt").write_text("\n".join(f"[{mmss(s['s'])}-{mmss(s['e'])}] {s['t']}" for s in segs),
                                          encoding="utf-8")
    where = sheets(video, [(s["s"] + 0.3, s["t"]) for s in segs], folder / "sheets")
    sheets(video, [(t, "") for t in np.arange(1, length, 10)], folder / "overview")
    smooth, hitches = playback_check(video)
    mapping, box = match(video, render) if render and render.exists() else (None, None)

    L = [f"# Review: {info.get('name', folder.name)}", ""]
    if info.get("mic"):
        L += [f"- Recorded {info.get('started', '')}, length {mmss(length)}; mic: {info.get('mic')}; "
              f"PC sound: {info.get('pc_sound')}",
              "- Video: review.mp4 (track 1 = mic + PC sound, track 2 = mic raw, track 3 = PC sound)."]
    else:
        L += [f"- Video: {video.name}, length {mmss(length)}."]
    L.append(f"- Playback check: {smooth}.")
    if hitches:
        L.append("  Hitches (recording time, length): " + ", ".join(f"{mmss(t)} ({g * 1000:.0f} ms)" for t, g in hitches[:30]))
    if info.get("heavy_jobs_during"):
        L.append(f"- A heavy job ran during the recording: {', '.join(info['heavy_jobs_during'])} "
                 f"(hitches near those times may come from that, not from the video).")
    if render:
        L.append(f"- Matched to: {render.name}" + (f" (player area {box})" if box else "") +
                 "; 'video' = that file's time, '-' = not on screen or no clear match.")
    L += ["- Sheets: sheets/NN.jpg (screen when each comment starts), overview/NN.jpg (every 10 s).", "",
          "| rec time | video | said | sheet |", "|---|---|---|---|"]
    for s, sh in zip(segs, where):
        v = at(mapping, s["s"] + 0.3) if mapping else None
        L.append(f"| {mmss(s['s'])} | {mmss(v) if v is not None else '-'} | {s['t'].replace('|', '/')} | {sh:02d} |")
    if not segs:
        L.append("| - | - | (no speech found) | - |")
    (folder / "REVIEW.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("REVIEW.md written:", folder / "REVIEW.md", flush=True)


if __name__ == "__main__":
    main()
