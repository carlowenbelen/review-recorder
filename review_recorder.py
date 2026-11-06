"""Review Recorder (Windows) - record a video review exactly as smooth as the video plays: the screen, your mic and
the PC sound as separate tracks, no time limit. Then read_review.py turns it into notes an AI (e.g. Claude) can read:
your words with times, frame sheets, a smoothness check and, with --render, the matching second of the reviewed video.

    python review_recorder.py          (or double-click "Review Recorder.bat")

Screen: ffmpeg ddagrab (Windows Desktop Duplication) on the GPU that drives the screen, straight into that GPU's
encoder (AMD AMF / NVIDIA NVENC / Intel QSV, picked automatically; CPU x264 as the last fallback). It records every
real screen change up to the screen's refresh rate, each with its true time, so a 30 or 60 fps video plays back as
smooth as it really was. Tested with frame-numbered clips playing full screen: 0 of 300 frames missed (30 fps) and
0 of 600 (60 fps), where a fixed 60 fps capture missed 5-13 %. The mouse is drawn. See tests/.
Audio: the mic is kept RAW (no denoise / EQ / compression); only a levelled temporary copy goes to Whisper.
Output: <output folder>/<date time> <name>/review.mp4 (+ raw tracks, info.json, REVIEW.md).
Settings (review_settings.json next to this file, all optional): "out_dir", "encoder" (amf / nvenc / qsv / x264),
"sync_fix" (seconds), "mic" (remembered automatically). Env HEAVY_LOCK = path of a JSON lock file
{"pid": .., "label": ..} written by your render queue, to warn while a render runs.
"""
import datetime as dt
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import ttk

HERE = Path(__file__).resolve().parent
if sys.executable.lower().endswith("pythonw.exe"):          # no black window: logs go to a file
    sys.stdout = sys.stderr = open(HERE / "review_console_log.txt", "a", encoding="utf-8", buffering=1)

import numpy as np                     # noqa: E402
import psutil                          # noqa: E402
import pyaudiowpatch as pa             # noqa: E402
import pystray                         # noqa: E402
from PIL import Image, ImageDraw       # noqa: E402

SETTINGS = HERE / "review_settings.json"


def settings(**new):
    try:
        s = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        s = {}
    if new:
        s.update(new)
        SETTINGS.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


OUT = Path(settings().get("out_dir") or (Path.home() / "Videos" / "Reviews"))
LOCK = Path(os.environ["HEAVY_LOCK"]) if os.environ.get("HEAVY_LOCK") else None
NO_WINDOW = 0x08000000
CHUNK = 1024
OPEN_FOLDER = True                      # show the finished review in Explorer (tests turn it off)
SYNC_FIX = float(settings().get("sync_fix", 0.124))   # measured with a flash + beep test clip (tests/): without it
                                        # the sound came out ~124 ms before the picture (ffmpeg reports its first frame late)
GRAB = "ddagrab=output_idx=0:framerate=144:dup_frames=0:draw_mouse=1"
ENCODERS = {    # GPU-side encoders take the captured frames without a copy to RAM; the others need one (more CPU)
    "amf": ("", ["-c:v", "h264_amf", "-usage", "lowlatency", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"]),
    "nvenc": (",hwdownload,format=bgra", ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "vbr",
                                          "-cq", "23", "-pix_fmt", "yuv420p"]),
    "qsv": (",hwmap=derive_device=qsv,format=qsv", ["-c:v", "h264_qsv", "-global_quality", "23"]),
    "x264": (",hwdownload,format=bgra", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p"]),
}


def pick_encoder():
    """The saved encoder, else the first one that records 1 s of the screen without an error (saved for next time)."""
    saved = settings().get("encoder")
    if saved in ENCODERS:
        return saved
    ff = shutil.which("ffmpeg") or "ffmpeg"
    for name, (vf, enc) in ENCODERS.items():
        r = subprocess.run([ff, "-v", "error", "-y", "-t", "1", "-f", "lavfi", "-i", GRAB + vf, *enc, "-f", "null", "-"],
                           capture_output=True, creationflags=NO_WINDOW)
        if r.returncode == 0:
            settings(encoder=name)
            return name
    return "x264"


def stamp(sec):
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


def db(x):
    return 20 * math.log10(x) if x > 1e-6 else -99.0


class Track:
    """Reads one audio device. With a path it also writes a WAV (the PC-sound track is padded with silence by the
    clock, because Windows sends no loopback data while nothing plays). peak = loudest sample of the last 0.3 s."""

    def __init__(self, p, device, path=None, loopback=False):
        self.p, self.dev, self.path, self.loopback = p, device, path, loopback
        self.ch = max(1, min(2, int(device["maxInputChannels"])))
        self.rate = int(device["defaultSampleRate"])
        self.stop = threading.Event()
        self.peak = 0.0
        self.t0 = None                   # perf_counter of the first sample written
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        wf = None
        try:
            stream = self.p.open(format=pa.paInt16, channels=self.ch, rate=self.rate, input=True,
                                 input_device_index=self.dev["index"], frames_per_buffer=CHUNK)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            return
        if self.path:
            wf = wave.open(str(self.path), "wb")
            wf.setnchannels(self.ch); wf.setsampwidth(2); wf.setframerate(self.rate)
        written, hold, hold_t = 0, 0.0, time.perf_counter()
        try:
            while not self.stop.is_set():
                now = time.perf_counter()
                if self.t0 is None:
                    self.t0 = now
                if stream.get_read_available() >= CHUNK:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    a = np.frombuffer(data, np.int16)
                    pk = float(np.abs(a).max()) / 32768 if a.size else 0.0
                    if pk >= hold or now - hold_t > 0.3:
                        hold, hold_t = pk, now
                    self.peak = hold
                    if wf:
                        wf.writeframes(data)
                    written += len(data) // (2 * self.ch)
                else:
                    time.sleep(0.005)
                    if self.loopback:
                        due = int((time.perf_counter() - self.t0) * self.rate) - written
                        if due > self.rate // 10:
                            if wf:
                                wf.writeframes(b"\x00" * (due * 2 * self.ch))
                            written += due
                            if now - hold_t > 0.3:
                                self.peak = hold = 0.0
        finally:
            stream.stop_stream(); stream.close()
            if wf:
                wf.close()

    def start(self):
        self.thread.start(); return self

    def finish(self):
        self.stop.set(); self.thread.join(timeout=5)


def mics(p):
    """Input devices, best first: the mic picked last time, then a headset / USB mic (a laptop's built-in
    'Microphone Array' sits far from your mouth = quiet voice), then the Windows default."""
    w = p.get_host_api_info_by_type(pa.paWASAPI)
    out = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d["hostApi"] == w["index"] and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice"):
            out.append(d)
    default = p.get_device_info_by_index(w["defaultInputDevice"])
    saved = settings().get("mic")

    def rank(d):
        if d["name"] == saved:
            return 0
        if not any(k in d["name"] for k in ("Array", "Broadcast", "Stereo Mix")):
            return 1
        return 2 if d["index"] == default["index"] else 3
    out.sort(key=rank)
    return out


def loopback_dev(p):
    w = p.get_host_api_info_by_type(pa.paWASAPI)
    spk = p.get_device_info_by_index(w["defaultOutputDevice"])
    for d in p.get_loopback_device_info_generator():
        if spk["name"] in d["name"]:
            return d
    return None


def render_running():
    """Label of the heavy job (e.g. a render) running now, from the optional HEAVY_LOCK file, or None."""
    if not LOCK:
        return None
    try:
        info = json.loads(LOCK.read_text(encoding="utf-8"))
        if psutil.pid_exists(int(info.get("pid", -1))):
            return info.get("label") or "a heavy job"
    except (OSError, ValueError):
        pass
    return None


def tray_image(recording):
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((4, 4, 60, 60), fill=(220, 30, 40, 255) if recording else (40, 90, 200, 255))
    d.rectangle((16, 20, 40, 44), outline=(255, 255, 255, 255), width=4)          # a little screen
    d.polygon([(42, 26), (52, 20), (52, 44), (42, 38)], fill=(255, 255, 255, 255))  # camera
    return im


class App:
    def __init__(self, root):
        self.root = root
        root.title("Review Recorder")
        root.geometry("600x520")
        root.minsize(560, 480)
        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass
        self.q = queue.Queue()
        self.rec = None
        self.closing = False
        self.p = pa.PyAudio()
        self.mic_list = mics(self.p)
        self.lb = loopback_dev(self.p)
        self.mon_mic = self.mon_pc = None

        f = ttk.Frame(root, padding=14); f.pack(fill="both", expand=True)
        ttk.Label(f, text="What are you reviewing?").grid(row=0, column=0, sticky="w")
        self.name = tk.StringVar(value="Review")
        ttk.Entry(f, textvariable=self.name, width=46).grid(row=0, column=1, sticky="we", padx=(8, 0))
        ttk.Label(f, text="Microphone").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.mic_name = tk.StringVar(value=self.mic_list[0]["name"] if self.mic_list else "")
        mf = ttk.Frame(f); mf.grid(row=1, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        self.cb = ttk.Combobox(mf, textvariable=self.mic_name, values=[d["name"] for d in self.mic_list], state="readonly", width=38)
        self.cb.pack(side="left", fill="x", expand=True)
        self.cb.bind("<<ComboboxSelected>>", lambda e: (settings(mic=self.mic_name.get()), self.monitor()))
        ttk.Button(mf, text="↻ Refresh", width=10, command=self.refresh_devices).pack(side="left", padx=(6, 0))
        ttk.Label(f, text="PC sound").grid(row=2, column=0, sticky="w")
        self.lbl_pc = ttk.Label(f, text=self.lb["name"].replace(" [Loopback]", "") if self.lb else "not found")
        self.lbl_pc.grid(row=2, column=1, sticky="w", padx=(8, 0))

        self.cv = tk.Canvas(f, height=58, bg="#1d1d22", highlightthickness=0)
        self.cv.grid(row=3, column=0, columnspan=2, sticky="we", pady=(10, 2))
        self.lbl_level = tk.Label(f, text="Say something to test your mic", fg="#555")
        self.lbl_level.grid(row=4, column=0, columnspan=2, sticky="w")

        self.v_hide = tk.BooleanVar(value=True)
        self.v_read = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Hide this window while recording (it stays out of the video)", variable=self.v_hide).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(f, text="After stopping, make the transcript + frame sheets for Claude", variable=self.v_read).grid(row=6, column=0, columnspan=2, sticky="w")

        b = ttk.Frame(f); b.grid(row=7, column=0, columnspan=2, sticky="w", pady=8)
        self.btn = ttk.Button(b, text="●  Start review", command=self.toggle, width=22); self.btn.pack(side="left")
        ttk.Button(b, text="Open reviews", command=lambda: os.startfile(OUT)).pack(side="left", padx=8)
        ttk.Button(b, text="Hide to tray", command=self.on_close).pack(side="left")
        self.lbl_time = tk.Label(f, text="00:00", font=("Segoe UI", 26, "bold")); self.lbl_time.grid(row=8, column=0, columnspan=2, sticky="w")
        self.lbl_warn = tk.Label(f, text="", fg="#b00020", wraplength=560, justify="left"); self.lbl_warn.grid(row=9, column=0, columnspan=2, sticky="w")
        self.log = tk.Text(f, height=6, wrap="word", relief="flat", bg="#f4f4f4"); self.log.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        f.columnconfigure(1, weight=1); f.rowconfigure(10, weight=1)
        OUT.mkdir(parents=True, exist_ok=True)
        self.say("Ready. Open the video in VLC, press Start, then play and talk.")
        if not self.lb:
            self.say("PC sound device not found - check Windows sound output.")
        self.monitor()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Unmap>", self.on_minimize)
        self.hint_shown = False
        self.tray_rec = None
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: "Stop review" if self.rec else "Start review", lambda: self.q.put(("tray", "toggle"))),
            pystray.MenuItem("Show window", lambda: self.q.put(("tray", "show")), default=True),
            pystray.MenuItem("Open reviews", lambda: os.startfile(OUT)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self.q.put(("tray", "quit"))))
        self.tray = pystray.Icon("review_recorder", tray_image(False), "Review Recorder", menu)
        threading.Thread(target=self.tray.run, daemon=True).start()
        self.last_warn_check = 0
        root.after(100, self.pump)

    # ---------- helpers
    def say(self, msg):
        self.log.insert("end", time.strftime("%H:%M ") + msg + "\n"); self.log.see("end")

    def mic_dev(self):
        return next((d for d in self.mic_list if d["name"] == self.mic_name.get()), self.mic_list[0] if self.mic_list else None)

    def refresh_devices(self, restart_monitor=True):
        """Look again for mics / speakers (e.g. a headset plugged in after the app opened)."""
        if self.rec:
            return
        for t in (self.mon_mic, self.mon_pc):
            if t:
                t.finish()
        self.mon_mic = self.mon_pc = None
        try:
            self.p.terminate()
        except Exception:  # noqa: BLE001
            pass
        self.p = pa.PyAudio()
        self.mic_list = mics(self.p)
        self.lb = loopback_dev(self.p)
        names = [d["name"] for d in self.mic_list]
        self.cb.config(values=names)
        best = names[0] if names else ""
        saved = settings().get("mic")
        self.mic_name.set(saved if saved in names else best)
        self.lbl_pc.config(text=self.lb["name"].replace(" [Loopback]", "") if self.lb else "not found")
        if restart_monitor:
            self.monitor()

    def monitor(self):
        """Live meters while not recording (nothing is saved)."""
        for t in (self.mon_mic, self.mon_pc):
            if t:
                t.finish()
        d = self.mic_dev()
        self.mon_mic = Track(self.p, d).start() if d else None
        self.mon_pc = Track(self.p, self.lb, loopback=True).start() if self.lb else None
        self.loud_hist = []

    def draw_meters(self, mic, pc):
        c = self.cv; c.delete("all")
        w = max(100, c.winfo_width()); bw = w - 110
        for row, (label, pk) in enumerate((("My mic", mic), ("PC sound", pc))):
            y = 8 + row * 26
            c.create_text(8, y + 8, text=label, fill="#ddd", anchor="w", font=("Segoe UI", 9))
            c.create_rectangle(90, y, 90 + bw, y + 16, fill="#2c2c33", width=0)
            v = max(0.0, min(1.0, (db(pk) + 60) / 60))
            col = "#3ccf6e" if db(pk) < -6 else ("#f2c94c" if db(pk) < -1 else "#eb5757")
            c.create_rectangle(90, y, 90 + bw * v, y + 16, fill=col, width=0)
            for mark in (-24, -12, -6):
                x = 90 + bw * (mark + 60) / 60
                c.create_line(x, y, x, y + 16, fill="#555")
            c.create_text(90 + bw + 4, y + 8, text=f"{db(pk):.0f} dB" if pk > 0 else "-", fill="#ddd", anchor="w", font=("Segoe UI", 9))

    def level_hint(self, mic_pk):
        """Judge the mic from the loud moments of the last ~6 s (speech peaks)."""
        self.loud_hist.append(mic_pk)
        self.loud_hist = self.loud_hist[-60:]
        top = sorted(self.loud_hist)[-10:]
        if max(self.loud_hist) < 0.003:
            return "My mic: no sound at all - pick another microphone above (or unmute it).", "#b00020"
        if db(max(top)) < -30:
            return "My mic: silent / very quiet - talk now to test it.", "#555"
        if db(max(top)) < -20:
            return "My mic: QUIET - move the mic closer to your mouth (raw voice is kept, so fix it at the mic).", "#b36b00"
        if db(max(top)) > -1:
            return "My mic: too loud (it clips) - move the mic a bit away.", "#b00020"
        return "My mic: good level.", "#1b7f3b"

    # ---------- loop
    def pump(self):
        while not self.q.empty():
            kind, val = self.q.get()
            if kind == "log":
                self.say(val)
            elif kind == "done":
                self.btn.config(state="normal", text="●  Start review")
                self.monitor()
                if self.closing:
                    self.tray.stop(); self.root.destroy(); return
            elif kind == "tray":
                if val == "toggle" and str(self.btn["state"]) == "normal":
                    self.toggle()
                elif val == "show":
                    self.root.deiconify(); self.root.state("normal"); self.root.lift(); self.root.focus_force()
                elif val == "quit":
                    self.quit()
        r = self.rec
        mic_t = (r.get("mic") if r else self.mon_mic)
        pc_t = (r.get("pc") if r else self.mon_pc)
        mic_pk = mic_t.peak if mic_t else 0.0
        pc_pk = pc_t.peak if pc_t else 0.0
        self.draw_meters(mic_pk, pc_pk)
        txt, col = self.level_hint(mic_pk)
        self.lbl_level.config(text=txt, fg=col)
        now = time.time()
        if now - self.last_warn_check > 3:
            self.last_warn_check = now
            job = render_running()
            ram = psutil.virtual_memory().available / 2**30
            warn = []
            if job:
                warn.append(f"A heavy job is running now ({job}). The recording can drop frames - if smoothness matters, wait for it to finish.")
            if ram < 0.7:
                warn.append(f"Only {ram:.1f} GB of memory free - close other screen recorders / extra browser tabs if the PC feels slow.")
            self.lbl_warn.config(text="\n".join(warn))
            if r and job and job not in r["jobs"]:
                r["jobs"].append(job)
        recording = bool(r)
        if recording != self.tray_rec:
            self.tray_rec = recording
            self.tray.icon = tray_image(recording)
            self.tray.update_menu()
        title = ("Review Recorder - recording " + stamp(time.perf_counter() - r["t0"])) if recording else "Review Recorder"
        if self.tray.title != title:
            self.tray.title = title
        if r:
            self.lbl_time.config(text=stamp(time.perf_counter() - r["t0"]))
            if r["ff"].poll() is not None and not r.get("ff_dead"):
                r["ff_dead"] = True
                self.say("The screen recorder stopped by itself: " + r.get("ff_err", "")[-200:])
        self.root.after(100, self.pump)

    def toggle(self):
        self.stop() if self.rec else self.start()

    # ---------- record
    def start(self):
        chosen = self.mic_name.get()
        self.refresh_devices(restart_monitor=False)           # the headset may have been plugged / unplugged
        if chosen in [d["name"] for d in self.mic_list]:
            self.mic_name.set(chosen)
        if not self.lb:
            self.say("No PC sound device - cannot start."); self.monitor(); return
        name = re.sub(r'[\\/:*?"<>|]+', " ", self.name.get().strip() or "Review").strip()
        folder = OUT / f"{dt.datetime.now():%Y-%m-%d %H%M} {name}"
        folder.mkdir(parents=True, exist_ok=True)
        for t in (self.mon_mic, self.mon_pc):
            if t:
                t.finish()
        self.mon_mic = self.mon_pc = None
        mic = self.mic_dev()
        rec = {"folder": folder, "name": name, "jobs": [], "mic_name": mic["name"] if mic else None,
               "pc_name": self.lb["name"], "first_frame": None}
        enc = pick_encoder()
        rec["encoder"] = enc
        cmd = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-f", "lavfi",
               "-i", GRAB + ENCODERS[enc][0], *ENCODERS[enc][1], "-fps_mode", "passthrough", "-progress", "pipe:1", "-stats_period", "0.05",
               str(folder / "screen.mp4")]
        rec["ff"] = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=NO_WINDOW)
        threading.Thread(target=self.watch_ffmpeg, args=(rec,), daemon=True).start()
        threading.Thread(target=self.watch_err, args=(rec,), daemon=True).start()
        rec["pc"] = Track(self.p, self.lb, folder / "pc.wav", loopback=True).start()
        if mic:
            rec["mic"] = Track(self.p, mic, folder / "mic.wav").start()
        rec["t0"] = time.perf_counter()
        self.rec = rec
        self.btn.config(text="■  Stop review")
        self.say(f"Recording: {folder.name}  (mic: {rec['mic_name']})")
        if self.v_hide.get():
            self.root.after(600, self.root.withdraw)

    def watch_ffmpeg(self, rec):
        """The first time ffmpeg reports a frame = when the video starts (lines the audio up with the picture)."""
        for line in rec["ff"].stdout:
            if rec["first_frame"] is None and line.startswith(b"frame="):
                try:
                    if int(line.split(b"=")[1]) >= 1:
                        rec["first_frame"] = time.perf_counter()
                except ValueError:
                    pass

    def watch_err(self, rec):
        rec["ff_err"] = ""
        for line in rec["ff"].stderr:
            rec["ff_err"] = (rec["ff_err"] + line.decode(errors="replace"))[-2000:]

    def stop(self):
        rec, self.rec = self.rec, None
        self.btn.config(state="disabled", text="Saving...")
        try:
            rec["ff"].stdin.write(b"q"); rec["ff"].stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        rec["t1"] = time.perf_counter()
        for k in ("pc", "mic"):
            if rec.get(k):
                rec[k].finish()
        try:
            rec["ff"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            rec["ff"].kill()
        threading.Thread(target=self.finish, args=(rec, self.v_read.get()), daemon=True).start()

    def finish(self, rec, read_after):
        log = lambda m: self.q.put(("log", m))   # noqa: E731
        folder = rec["folder"]
        ff = shutil.which("ffmpeg") or "ffmpeg"
        try:
            v0 = rec["first_frame"] or rec["t0"]
            info = {"name": rec["name"], "started": folder.name[:15], "mic": rec["mic_name"], "pc_sound": rec["pc_name"],
                    "screen": "ddagrab 144 max, real changes only (VFR), " + rec.get("encoder", "?"), "heavy_jobs_during": rec["jobs"],
                    "length_s": round(rec["t1"] - v0, 1)}
            raw = []
            for k, label in (("mic", "my mic (raw).flac"), ("pc", "pc sound (raw).flac")):
                t = rec.get(k)
                if t and (folder / f"{k}.wav").exists() and t.t0:
                    off = v0 - t.t0 - SYNC_FIX          # >0: the audio started before the first picture
                    info[f"{k}_offset_s"] = round(off, 3)
                    trim = ["-af", f"atrim=start={off:.4f},asetpts=PTS-STARTPTS"] if off > 0 else \
                           ["-af", f"adelay={int(-off * 1000)}:all=1"]
                    subprocess.run([ff, "-v", "error", "-y", "-i", str(folder / f"{k}.wav"), *trim, "-c:a", "flac",
                                    str(folder / label)], creationflags=NO_WINDOW, check=True)
                    raw.append(folder / label)
            if not (folder / "screen.mp4").exists():
                raise RuntimeError("no screen video: " + rec.get("ff_err", "")[-300:])
            ins = ["-i", str(folder / "screen.mp4")]
            for f in raw:
                ins += ["-i", str(f)]
            maps, fc = ["-map", "0:v"], ""
            if len(raw) == 2:                         # track 1 = both (plain mix, no filters), 2 = my mic, 3 = PC sound
                fc = "[1:a]aformat=channel_layouts=stereo[m];[2:a]aformat=channel_layouts=stereo[p];[m][p]amix=inputs=2:normalize=0:duration=longest[mix]"
                maps += ["-map", "[mix]", "-map", "1:a", "-map", "2:a",
                         "-metadata:s:a:0", "title=Mic + PC sound", "-metadata:s:a:1", "title=My mic (raw)",
                         "-metadata:s:a:2", "title=PC sound"]
            elif raw:
                maps += ["-map", "1:a"]
            cmd = [ff, "-v", "error", "-y", *ins]
            if fc:
                cmd += ["-filter_complex", fc]
            cmd += [*maps, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(folder / "review.mp4")]
            subprocess.run(cmd, creationflags=NO_WINDOW, check=True)
            (folder / "screen.mp4").unlink()
            for k in ("mic.wav", "pc.wav"):
                (folder / k).unlink(missing_ok=True)
            (folder / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            mb = (folder / "review.mp4").stat().st_size / 2**20
            log(f"Saved review.mp4 ({stamp(info['length_s'])}, {mb:.0f} MB).")
            if read_after:
                py = Path(sys.executable).with_name("python.exe")
                subprocess.Popen([str(py), "-X", "utf8", str(HERE / "read_review.py"), str(folder)],
                                 stdout=open(folder / "read_log.txt", "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                                 creationflags=NO_WINDOW)
                log("Making the transcript + sheets in the background (REVIEW.md when done).")
        except Exception as e:  # noqa: BLE001
            log(f"Could not finish: {e}")
        self.q.put(("done", None))
        if OPEN_FOLDER:
            subprocess.Popen(["explorer.exe", "/select,", str(folder / "review.mp4")])

    # ---------- window / tray
    def on_close(self):
        self.root.withdraw()
        if not self.hint_shown:
            self.hint_shown = True
            try:
                self.tray.notify("Still running in the hidden icons. Right-click it to start/stop, show or quit.", "Review Recorder")
            except Exception:  # noqa: BLE001
                pass

    def on_minimize(self, event):
        if event.widget is self.root and self.root.state() == "iconic":
            self.root.after(10, self.on_close)

    def quit(self):
        if self.rec or str(self.btn["state"]) == "disabled":
            self.closing = True
            if self.rec:
                self.stop()
            self.say("Saving first - the app will close by itself.")
        else:
            for t in (self.mon_mic, self.mon_pc):
                if t:
                    t.finish()
            self.tray.stop(); self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
