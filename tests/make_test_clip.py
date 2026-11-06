"""Test clips for the Review Recorder: every frame carries its own number as two rows of 16 black/white blocks
(top + bottom, so a torn frame shows up), four red corner marks (to find the picture inside a player window),
a moving bar, a big frame number, and a white flash + 1 kHz beep on every whole second (audio/video sync check).

    python -X utf8 make_test_clip.py            -> test30.mp4 + test60.mp4 (12 s, 1920x1080)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1920, 1080
BITS = 16
BLK, GAP = 64, 16
X0 = (W - BITS * (BLK + GAP) + GAP) // 2
ROWS_Y = (80, H - 80 - BLK)          # top row, bottom row (top-left y of the blocks)
FID = 48                              # red corner marks: 48 px squares, 16 px in from each corner
FID_IN = 16
FLASH = (120, 440, 200, 200)          # x, y, w, h of the sync flash square


def code(n):
    """12 bits frame number + 4 bits check (number of 1-bits mod 16)."""
    n &= 0xFFF
    return n | ((bin(n).count("1") % 16) << 12)


def block_centers():
    """(x, y) centre of every bit block, row by row: row 0 = top, row 1 = bottom."""
    return [[(X0 + i * (BLK + GAP) + BLK // 2, y + BLK // 2) for i in range(BITS)] for y in ROWS_Y]


def fid_centers():
    c = FID_IN + FID // 2
    return [(c, c), (W - c, c), (c, H - c), (W - c, H - c)]     # TL, TR, BL, BR


def frame(n, fps, font):
    im = Image.new("RGB", (W, H), (40, 40, 44))
    d = ImageDraw.Draw(im)
    for (x, y) in fid_centers():
        d.rectangle((x - FID // 2, y - FID // 2, x + FID // 2, y + FID // 2), fill=(255, 0, 0))
    v = code(n)
    for row in block_centers():
        for i, (x, y) in enumerate(row):
            on = (v >> i) & 1
            d.rectangle((x - BLK // 2, y - BLK // 2, x + BLK // 2, y + BLK // 2), fill=(250, 250, 250) if on else (5, 5, 5))
    bx = int((n * (W - 40) / (2 * fps)) % (W - 40))            # bar crosses the screen every 2 s
    d.rectangle((bx, 200, bx + 40, 880), fill=(255, 210, 0))
    if n % fps < 3:                                              # 3-frame white flash on every whole second
        x, y, w, h = FLASH
        d.rectangle((x, y, x + w, y + h), fill=(255, 255, 255))
    d.text((W // 2, H // 2), f"{n}", fill=(230, 230, 230), font=font, anchor="mm")
    d.text((W // 2, H // 2 + 150), f"{fps} fps  {n / fps:6.2f} s", fill=(160, 160, 160), font=font.font_variant(size=48), anchor="mm")
    return im.tobytes()


def beeps(fps, secs, rate=48000):
    t = np.arange(int(secs * rate)) / rate
    a = np.zeros_like(t)
    for s in range(int(secs) + 1):
        m = (t >= s) & (t < s + 0.08)
        a[m] = 0.5 * np.sin(2 * np.pi * 1000 * (t[m] - s))
    return (a * 32767).astype(np.int16)


def make(fps, secs=12):
    out = HERE / f"test{fps}.mp4"
    wav = HERE / f"beeps{fps}.wav"
    import wave
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000); wf.writeframes(beeps(fps, secs).tobytes())
    try:
        font = ImageFont.truetype("arialbd.ttf", 220)
    except OSError:
        font = ImageFont.load_default()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "fast", "-crf", "16",
                          "-pix_fmt", "yuv420p", "-g", str(fps), "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)],
                         stdin=subprocess.PIPE)
    for n in range(int(fps * secs)):
        p.stdin.write(frame(n, fps, font))
    p.stdin.close(); p.wait()
    wav.unlink()
    print("made", out.name)


if __name__ == "__main__":
    for f in [int(a) for a in sys.argv[1:]] or (30, 60):
        make(f)
