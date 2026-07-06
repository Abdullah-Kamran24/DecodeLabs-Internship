#!/usr/bin/env python3
"""
demo_preprocessing_stages.py
------------------------------
"The Logic Skeleton: Systematic Image Pre-Processing" — visualized.

This demo script saves EACH intermediate stage of the pre-processing
pipeline as its own image file, so you can visually inspect exactly
what "Grayscale Conversion", "Gaussian Blur", "Deskewing", and
"Adaptive Thresholding" do to a raw input — the same four panels shown
in the Project 4 briefing deck.

Usage:
    python demo_preprocessing_stages.py [image_path]

If no image_path is given, sample_images/sample_invoice.png is used.
Output is written to output/stages/<name>_<step>.png
"""

from __future__ import annotations

import os
import sys

import cv2

from core.preprocessing import (
    to_grayscale,
    gaussian_blur,
    deskew,
    adaptive_threshold,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "stages")


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_images/sample_invoice.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image at '{image_path}'.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Running the Logic Skeleton pipeline on: {image_path}\n")

    # Stage 0: original
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_0_original.png"), image)
    print("[0] Original             -> saved")

    # Stage 1: Grayscale Conversion
    gray = to_grayscale(image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_1_grayscale.png"), gray)
    print("[1] Grayscale Conversion -> saved  (3D RGB matrix -> 1D intensity matrix)")

    # Stage 2: Gaussian Blur
    blurred = gaussian_blur(gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_2_gaussian_blur.png"), blurred)
    print("[2] Gaussian Blur        -> saved  (micro-imperfections / noise smoothed)")

    # Stage 3: Deskewing
    deskewed, angle = deskew(blurred)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_3_deskewed.png"), deskewed)
    print(f"[3] Deskewing            -> saved  (rotation applied: {angle:.3f} degrees)")

    # Stage 4: Adaptive Thresholding (Otsu's Method)
    binary, cutoff = adaptive_threshold(deskewed)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_4_binary_threshold.png"), binary)
    print(f"[4] Adaptive Thresholding-> saved  (Otsu cutoff = {cutoff:.1f})")
    print(f"      IF pixel_intensity >= {cutoff:.0f}: pixel = 255 (white)")
    print(f"      IF pixel_intensity <  {cutoff:.0f}: pixel = 0   (black)")

    print(f"\nAll stages saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
