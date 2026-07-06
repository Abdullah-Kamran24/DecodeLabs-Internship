"""
models/download_models.py
---------------------------
"Transfer Learning: Inheriting the Machine's Knowledge"

Why train an AI from scratch when you can download a degree?

    The Base:     We leverage a pre-trained model (MobileNet-SSD) that has
                  already analyzed millions of ImageNet images to
                  understand universal visual concepts (edges, shapes,
                  gradients).
    The Transfer: We detach the final output layer and plug in our own
                  Object Detection task.

This script fetches the two files Path 2 (Object Detection) needs:

    MobileNetSSD_deploy.prototxt    -> the network architecture definition
    MobileNetSSD_deploy.caffemodel  -> the pre-trained weights (~23 MB)

Run this once before using Path 2:
    python models/download_models.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "MobileNetSSD_deploy.prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    ),
    "MobileNetSSD_deploy.caffemodel": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
        "mobilenet_iter_73000.caffemodel"
    ),
}

MIN_EXPECTED_SIZE = {
    "MobileNetSSD_deploy.prototxt": 10_000,        # ~29 KB actual
    "MobileNetSSD_deploy.caffemodel": 20_000_000,  # ~23 MB actual
}


def _download(filename: str, url: str) -> None:
    dest_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) >= MIN_EXPECTED_SIZE[filename]:
        print(f"[SKIP] {filename} already present ({os.path.getsize(dest_path):,} bytes).")
        return

    print(f"[DOWNLOAD] {filename} <- {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to download {filename}: {exc}", file=sys.stderr)
        sys.exit(1)

    size = os.path.getsize(dest_path)
    if size < MIN_EXPECTED_SIZE[filename]:
        print(
            f"[ERROR] {filename} downloaded but looks too small "
            f"({size:,} bytes) — the file may be corrupted or the source "
            "may be unreachable from your network.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[OK] {filename} saved ({size:,} bytes).")


def main() -> None:
    print("=" * 70)
    print(" Project 4 — Downloading MobileNet-SSD (Path 2: Object Detection)")
    print("=" * 70)
    for filename, url in FILES.items():
        _download(filename, url)
    print("\nAll model files ready. You can now run Path 2 (Object Detection).")


if __name__ == "__main__":
    main()
