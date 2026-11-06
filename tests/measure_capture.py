"""Measure a screen recording of test30.mp4 / test60.mp4 (made by make_test_clip.py).

For every recorded frame it reads which clip frame was on screen (the 16-block code, top and bottom rows), then
reports, numbers only:
  - recording: frames, real fps, timing gaps (a gap = more than 1.5x the normal step)
  - clip frames seen: how many of the clip's frames reached the recording at least once (missing = judder)
  - torn frames (top and bottom rows disagree) and frames that could not be read
  - step between clip frames as the recording shows them (ms): average, spread, worst

    python -X utf8 measure_capture.py rec.mp4 30 [--json out.json]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_clip import BITS, block_centers, fid_centers, W as SW, H as SH  # noqa: E402

SCALE = 2          # read at half size (960x540): blocks are still 32 px


def probe_pts(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return np.array([float(x.strip(",")) for x in r.stdout.split() if x.strip(",").strip() and x.strip(",") != "N/A"])


def size(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def find_map(path, w, h):
    """Find the four red corner marks in a frame that shows the clip; return (sx, sy, ox, oy) clip -> recording."""
    for t in (3, 5, 7, 2, 9):
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        if len(r.stdout) < w * h * 3:
            continue
        a = np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).astype(int)
        red = (a[..., 0] > 170) & (a[..., 1] < 70) & (a[..., 2] < 70)
        ys, xs = np.nonzero(red)
        if len(xs) < 100:
            continue
        mx, my = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2      # corners around the clip, wherever it is
        pts = []
        for qy in (0, 1):
            for qx in (0, 1):
                m = ((xs > mx) == bool(qx)) & ((ys > my) == bool(qy))
                if m.sum() < 20:
                    break
                pts.append((xs[m].mean(), ys[m].mean()))
        if len(pts) == 4:
            src = fid_centers()            # TL, TR, BL, BR
            sx = (pts[1][0] - pts[0][0]) / (src[1][0] - src[0][0])
            sy = (pts[2][1] - pts[0][1]) / (src[2][1] - src[0][1])
            ox = pts[0][0] - src[0][0] * sx
            oy = pts[0][1] - src[0][1] * sy
            return sx, sy, ox, oy
    return None


def decode_all(path, w, h, mp):
    sx, sy, ox, oy = mp
    rw, rh = w // SCALE, h // SCALE
    rows = [[(int((x * sx + ox) / SCALE), int((y * sy + oy) / SCALE)) for x, y in row] for row in block_centers()]
    k = max(2, int(10 * sx / SCALE))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={rw}:{rh},format=gray",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    out = []
    n = rw * rh
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        a = np.frombuffer(buf, np.uint8).reshape(rh, rw)
        vals = []
        for row in rows:
            v = 0
            for i, (x, y) in enumerate(row):
                if a[y - k:y + k, x - k:x + k].mean() > 128:
                    v |= 1 << i
            vals.append(v)
        top, bot = vals

        def ok(v):
            return ((v >> 12) & 0xF) == bin(v & 0xFFF).count("1") % 16
        if top == bot and ok(top):
            out.append(top & 0xFFF)
        elif ok(top) and ok(bot):
            out.append(-2)             # torn: two different frames in one picture
        else:
            out.append(-1)             # not the clip / unreadable
    p.wait()
    return out


def measure(path, src_fps):
    path = Path(path)
    w, h = size(path)
    pts = probe_pts(path)
    mp = find_map(path, w, h)
    res = {"file": path.name, "src_fps": src_fps, "size": f"{w}x{h}", "frames": int(len(pts))}
    if len(pts) > 1:
        d = np.diff(pts)
        med = float(np.median(d))
        res.update(duration=round(float(pts[-1] - pts[0]), 2), fps=round((len(pts) - 1) / float(pts[-1] - pts[0]), 2),
                   step_ms=round(med * 1000, 2), gaps=int((d > 1.5 * med).sum()), max_gap_ms=round(float(d.max()) * 1000, 1))
    if not mp:
        res["error"] = "red corner marks not found (clip not on screen?)"
        return res
    codes = decode_all(path, w, h, mp)
    m = min(len(codes), len(pts))
    codes, pts = np.array(codes[:m]), pts[:m]
    good = codes >= 0
    res["torn"] = int((codes == -2).sum())
    res["unreadable"] = int((codes == -1).sum())
    if good.sum() < 10:
        res["error"] = "clip frames not readable"
        return res
    seen = codes[good]
    lo, hi = int(seen.min()), int(seen.max())
    # ignore the very first and last second (player opening / closing)
    lo2, hi2 = lo + src_fps, hi - src_fps
    inside = seen[(seen >= lo2) & (seen <= hi2)]
    want = set(range(lo2, hi2 + 1))
    got = set(int(x) for x in inside)
    missing = sorted(want - got)
    res.update(clip_range=f"{lo}-{hi}", checked=len(want), missing=len(missing),
               missing_pct=round(100 * len(missing) / max(1, len(want)), 2), missing_list=missing[:40])
    # when did each clip frame first appear in the recording -> step between clip frames
    first = {}
    for c, t in zip(codes, pts):
        if c >= lo2 and c <= hi2 and c not in first:
            first[int(c)] = float(t)
    ks = sorted(first)
    steps = np.array([(first[b] - first[a]) / (b - a) for a, b in zip(ks, ks[1:]) if b > a]) * 1000
    back = int(sum(1 for a, b in zip(codes[good], codes[good][1:]) if b < a and b >= lo2))
    if len(steps):
        ideal = 1000 / src_fps
        res.update(clip_step_ms=round(float(steps.mean()), 2), clip_step_sd=round(float(steps.std()), 2),
                   clip_step_worst=round(float(steps.max()), 1),
                   long_holds=int((steps > 1.6 * ideal).sum()), went_back=back)
    return res


if __name__ == "__main__":
    r = measure(sys.argv[1], int(sys.argv[2]))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
