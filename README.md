<h1 align="center">Hi, I'm Jay Patel 👋</h1>

<p align="center">
  Software engineer building systems tooling, P2P apps, and quant/trading infrastructure.
</p>

<p align="center">
  <em>AI-native builder — even in a stack or language I haven't used, I ramp fast and ship a working prototype, then production-ready code, quickly by pairing with AI/LLM tools (Claude Code, Gemini CLI, Codex, OpenCode).</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white" />
  <img src="https://img.shields.io/badge/Go-00ADD8?style=flat&logo=go&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white" />
</p>

---

### 🔭 What I work on
- **Systems & distributed tooling** — schedulers, daemons, P2P networking
- **Secure file transfer & messaging** — end-to-end encrypted, server-less
- **Quant & trading infrastructure** — market data pipelines, backtesting, analytics

### 🚀 Featured projects

| Project | What it is | Stack |
|---|---|---|
| [**rusty-sched**](https://github.com/jdp5949/rusty-sched) | Autosys OSS clone — single binary, SQLite/Postgres, conditions DSL, JIL import, live logs, retries, alerts, Web UI. Zero deps. | Rust |
| [**discord-alert-transport**](https://github.com/jdp5949/discord-alert-transport) | Portable Discord webhook transport — Python package + 10-language reference. Severity-routed alerts, 429 backoff, audit mirror. | Python |
| [**p2p-messaging**](https://github.com/jdp5949/p2p-messaging) | Peer-to-peer encrypted messaging without central servers. | Go |
| [**tg-ringer**](https://github.com/jdp5949/tg-ringer) | Ring (call) & message any Telegram user from your own account (userbot) — urgent alerts via a real Telegram call. PyPI package + CLI. | Python |

### 🛠️ Tech
`Rust` · `Go` · `Python` · `TypeScript` · `SQLite` · `PostgreSQL` · `Wails` · `React`

### 📫 Reach me
[![Email](https://img.shields.io/badge/Email-jvp0177%40gmail.com-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:jvp0177@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-jdp5949-181717?style=flat&logo=github&logoColor=white)](https://github.com/jdp5949)

<p align="center">
  <img src="https://github-readme-stats-jdp.vercel.app/api?username=jdp5949&show_icons=true&hide_rank=true&theme=default" height="160" />
  <img src="https://github-readme-stats-jdp.vercel.app/api/top-langs/?username=jdp5949&layout=compact" height="160" />
</p>



"""
Standalone QR file receiver — wire-compatible with qr_transfer.sender.

Usage:
  python mini_receiver.py <output_dir>                       # live webcam
  python mini_receiver.py <output_dir> --camera 1            # pick camera index
  python mini_receiver.py <output_dir> --input-dir ./frames  # decode PNGs (no cam)

Dependencies: opencv-python, pyzbar, zstandard, Pillow, numpy
  pip install opencv-python pyzbar zstandard Pillow numpy
  Windows: pyzbar wheel bundles zbar (nothing else to do).
  macOS:   brew install zbar ; run with DYLD_LIBRARY_PATH=/opt/homebrew/lib
"""

import argparse
import base64
import functools
import glob
import io
import math
import os
import random
import struct
import sys
import tarfile

import cv2
import numpy as np
import zstandard
from pyzbar import pyzbar

# ---------- wire format (must match sender exactly) ----------

MAGIC = b"\x51\x52"
HEADER_SIZE = 14


def unpack(data):
    """data -> (session_id, packet_id, k, payload) or None."""
    if len(data) < HEADER_SIZE or data[:2] != MAGIC:
        return None
    session_id = data[2:6]
    packet_id, k = struct.unpack(">II", data[6:14])
    return session_id, packet_id, k, data[14:]


# ---------- LT fountain decode (robust soliton; Python random.Random) ----------

@functools.lru_cache(maxsize=None)
def _degree_cdf(k):
    if k == 1:
        return ((1, 1.0),)
    c, delta = 0.1, 0.5
    R = c * math.log(k / delta) * math.sqrt(k)
    rho = [0.0] * (k + 1)
    rho[1] = 1.0 / k
    for d in range(2, k + 1):
        rho[d] = 1.0 / (d * (d - 1))
    tau = [0.0] * (k + 1)
    kr = max(1, min(k, int(round(k / R))))
    for d in range(1, kr):
        tau[d] = R / (d * k)
    tau[kr] = R * math.log(R / delta) / k
    w = [rho[d] + tau[d] for d in range(k + 1)]
    z = sum(w)
    cdf = []
    cum = 0.0
    for d in range(1, k + 1):
        cum += w[d] / z
        cdf.append((d, cum))
    return tuple(cdf)


def _soliton_degree(k, packet_id):
    r = random.Random(packet_id ^ 0xDEADBEEF).random()
    for d, cum in _degree_cdf(k):
        if r <= cum:
            return d
    return k


def _neighbors(k, degree, packet_id):
    return random.Random(packet_id ^ 0xCAFEBABE).sample(range(k), min(degree, k))


def decode(packets, k):
    """packets: list of (packet_id, payload). Returns k chunks or None."""
    if not packets:
        return None
    size = len(packets[0][1])
    nodes = []
    for pid, payload in packets:
        deg = _soliton_degree(k, pid)
        nbrs = set(_neighbors(k, deg, pid))
        nodes.append([nbrs, bytearray(payload)])
    recovered = [None] * k
    changed = True
    while changed:
        changed = False
        for node in nodes:
            nbrs, payload = node
            if len(nbrs) != 1:
                continue
            idx = next(iter(nbrs))
            if recovered[idx] is not None:
                node[0].clear()
                continue
            recovered[idx] = bytes(payload)
            node[0].clear()
            changed = True
            for other in nodes:
                if other is node:
                    continue
                if idx in other[0]:
                    other[0].discard(idx)
                    ob = other[1]
                    for j in range(size):
                        ob[j] ^= payload[j]
    if any(r is None for r in recovered):
        return None
    return recovered


# ---------- decompress + extract ----------

def extract(blob, outpath):
    os.makedirs(outpath, exist_ok=True)
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(blob)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        tar.extractall(outpath, filter="data")


# ---------- QR detection (robust + zoom preview) ----------

def detect_qrs(frame_bgr):
    """Try hard to find/decode QR codes in a BGR frame.
    Returns list of (raw_packet_bytes, polygon_points)."""
    results = []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Try several preprocessings — webcams vary a lot.
    candidates = [
        gray,
        cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]
    seen = set()
    for img in candidates:
        for r in pyzbar.decode(img):
            if r.type != "QRCODE":
                continue
            key = bytes(r.data)
            if key in seen:
                continue
            seen.add(key)
            try:
                raw = base64.b64decode(r.data)
            except Exception:
                continue
            pts = np.array([[p.x, p.y] for p in r.polygon], dtype=np.int32) if r.polygon else None
            results.append((raw, pts))
        if results:
            break  # got something at this scale, good enough
    return results


def draw_overlay(frame, polys, status):
    h, w = frame.shape[:2]
    for pts in polys:
        if pts is not None and len(pts) >= 4:
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def make_zoom(frame, pts, zoom_size=320):
    """Crop the detected QR region and upscale it into a preview tile."""
    if pts is None or len(pts) < 4:
        return None
    x, y, bw, bh = cv2.boundingRect(pts)
    pad = int(0.15 * max(bw, bh)) + 5
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + bw + pad); y1 = min(frame.shape[0], y + bh + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)


# ---------- main receive loop ----------

def run(output_dir, camera=0, input_dir=None):
    seen_ids = set()
    packets = []
    session_id = None
    k = None

    def feed(raw):
        nonlocal session_id, k
        r = unpack(raw)
        if r is None:
            return None
        sid, pid, pkt_k, payload = r
        if session_id is not None and sid != session_id:
            seen_ids.clear(); packets.clear()
            session_id = None
        if session_id is None:
            session_id = sid; k = pkt_k
            print(f"\nSession {sid.hex()} | K={k} chunks")
        if pid not in seen_ids:
            seen_ids.add(pid)
            packets.append((pid, payload))
            print(f"\r  {len(seen_ids)}/{k} unique packets", end="", flush=True)
        if len(seen_ids) >= k:
            chunks = decode(packets, k)
            if chunks is not None:
                return chunks
        return None

    # ----- file mode (test, no camera) -----
    if input_dir:
        from PIL import Image
        for p in sorted(glob.glob(os.path.join(input_dir, "*.png"))):
            for r in pyzbar.decode(Image.open(p)):
                if r.type != "QRCODE":
                    continue
                try:
                    raw = base64.b64decode(r.data)
                except Exception:
                    continue
                chunks = feed(raw)
                if chunks is not None:
                    print("\nDecoded. Extracting ...")
                    extract(b"".join(chunks), output_dir)
                    print(f"Done -> {output_dir}")
                    return
        print("\nFailed: ran out of frames.")
        return

    # ----- live webcam -----
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Cannot open camera {camera}. Try --camera 1 or 2.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera open. Fill the view with the QR. Press Q/ESC to quit.")

    last_zoom = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed"); break

        found = detect_qrs(frame)
        polys = [pts for _, pts in found]
        for raw, pts in found:
            z = make_zoom(frame, pts)
            if z is not None:
                last_zoom = z
            chunks = feed(raw)
            if chunks is not None:
                print("\nDecoded. Extracting ...")
                extract(b"".join(chunks), output_dir)
                print(f"Done -> {output_dir}")
                cap.release(); cv2.destroyAllWindows()
                return

        status = (f"session {session_id.hex()}  {len(seen_ids)}/{k}"
                  if session_id else "searching for QR ...")
        draw_overlay(frame, polys, status)
        if last_zoom is not None:
            frame[40:40 + last_zoom.shape[0], 8:8 + last_zoom.shape[1]] = last_zoom
        cv2.imshow("QR Receiver", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--input-dir")
    a = ap.parse_args()
    run(a.output_dir, camera=a.camera, input_dir=a.input_dir)


if __name__ == "__main__":
    main()
