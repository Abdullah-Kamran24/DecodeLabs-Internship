#!/usr/bin/env python3
"""
main.py
--------
Project 4: Building the Machine's Optic Nerve
The DecodeLabs Architect's Playbook for Image & Text Recognition

Objective:
    Engineer a Python script capable of ingesting raw visual data and
    extracting accurate, machine-readable intelligence.

The Deliverable:
    A fully functioning recognition pipeline that proves the machine can
    see text or objects with validated confidence.

This is the single entry point for both recognition paths:

    Path 1: OCR              (pytesseract)          -> text extraction
    Path 2: Object Detection (cv2.dnn + MobileNet-SSD) -> entity localization

Usage
-----
Interactive menu:
    python main.py

Direct CLI (no menu):
    python main.py --path ocr    --image sample_images/sample_invoice.png --psm 11
    python main.py --path detect --image sample_images/sample_object_dog_bike_car.jpg
    python main.py --path both   --image sample_images/sample_invoice.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"E:\Program Files\Tesseract-OCR\tesseract.exe"

from core.ocr_engine import (
    recognize_text, PSM_MODES, DEFAULT_PSM, TesseractNotConfiguredError,
)
from core.object_detector import detect_objects, load_model

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _banner() -> None:
    print("=" * 72)
    print(" PROJECT 4: BUILDING THE MACHINE'S OPTIC NERVE")
    print(" The DecodeLabs Architect's Playbook for Image & Text Recognition")
    print("=" * 72)


def _print_gatekeeper_summary(passed: bool, best_conf: float, kept: int, dropped: int) -> None:
    print("\n--- Gatekeeper Rule: Milestone Validation ---")
    status = "PASSED" if passed else "FAILED"
    print(f"  [3] Accuracy Benchmarking : {status}  "
          f"(best confidence = {best_conf * 100:.2f}%, minimum required = 80.00%)")
    print(f"  [4] Visual Confirmation   : {'PASSED' if kept > 0 else 'FAILED'}  "
          f"({kept} kept / {dropped} dropped by the 80% gate)")


def run_ocr(image_path: str, psm: int, lang: str, no_deskew: bool) -> bool:
    """Runs Path 1 end-to-end. Returns True on success, False if a
    handled/expected error occurred (input file missing, Tesseract
    engine missing, etc.) so callers can decide whether to continue."""
    print(f"\n>>> PATH 1: Optical Character Recognition (OCR)")
    print(f"    Image : {image_path}")
    print(f"    PSM   : {psm} — {PSM_MODES.get(psm, 'unknown')}")

    if not os.path.isfile(image_path):
        print(f"\n[ERROR] Image file not found: '{image_path}'.\n"
              f"        Check the path and try again.", file=sys.stderr)
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"\n[ERROR] '{image_path}' exists but could not be decoded as an "
              f"image.\n        Make sure it's a valid JPG/PNG/BMP file.",
              file=sys.stderr)
        return False

    try:
        result = recognize_text(image, psm=psm, lang=lang, apply_deskew=not no_deskew)
    except TesseractNotConfiguredError as exc:
        print(str(exc), file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        print(f"\n[ERROR] Path 1 (OCR) failed unexpectedly: {exc}", file=sys.stderr)
        return False

    print("\n--- Pre-Processing Integrity Report ---")
    for key, value in result.preprocessing_report.as_dict().items():
        print(f"  {key:22s}: {value}")

    print("\n--- Recognized Text (kept detections only, confidence >= 80%) ---")
    if result.kept_detections:
        for det in result.kept_detections:
            print(f"  '{det.label}'  -> {det.confidence_pct}%")
    else:
        print("  (No words passed the 80% confidence gate.)")

    print(f"\n--- Full reconstructed string ---\n  {result.full_text}")

    _print_gatekeeper_summary(
        result.gatekeeper_passed, result.best_confidence,
        len(result.kept_detections), len(result.dropped_detections),
    )

    _ensure_output_dir()
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_ocr_annotated.png")
    cv2.imwrite(out_path, result.annotated_image)
    print(f"\n[SAVED] Visual confirmation image -> {out_path}")
    return True


def run_detection(image_path: str) -> bool:
    """Runs Path 2 end-to-end. Returns True on success, False if a
    handled/expected error occurred (input file missing, model files
    missing, etc.) so callers can decide whether to continue."""
    print(f"\n>>> PATH 2: Object Detection (MobileNet-SSD)")
    print(f"    Image : {image_path}")

    if not os.path.isfile(image_path):
        print(f"\n[ERROR] Image file not found: '{image_path}'.\n"
              f"        Check the path and try again.", file=sys.stderr)
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"\n[ERROR] '{image_path}' exists but could not be decoded as an "
              f"image.\n        Make sure it's a valid JPG/PNG/BMP file.",
              file=sys.stderr)
        return False

    try:
        net = load_model()
    except FileNotFoundError as exc:
        print(
            f"\n[ERROR] {exc}\n"
            f"        Fix: run  python models/download_models.py",
            file=sys.stderr,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Failed to load the MobileNet-SSD model: {exc}", file=sys.stderr)
        return False

    try:
        result = detect_objects(image, net)
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        print(f"\n[ERROR] Path 2 (Object Detection) failed unexpectedly: {exc}",
              file=sys.stderr)
        return False

    print(f"\n--- Blob Construction ---")
    print(f"  4D Blob shape (N, C, H, W): {result.blob_shape}")

    print("\n--- Detected Objects (confidence >= 80%) ---")
    if result.kept_detections:
        for det in result.kept_detections:
            x, y, w, h = det.box
            print(f"  {det.label:12s} conf={det.confidence_pct:6.2f}%  box=(x={x}, y={y}, w={w}, h={h})")
    else:
        print("  (No objects passed the 80% confidence gate.)")

    if result.dropped_detections:
        print("\n--- Dropped (below 80% gate) ---")
        for det in result.dropped_detections:
            print(f"  {det.label:12s} conf={det.confidence_pct:6.2f}%  [DROPPED]")

    _print_gatekeeper_summary(
        result.gatekeeper_passed, result.best_confidence,
        len(result.kept_detections), len(result.dropped_detections),
    )

    _ensure_output_dir()
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_detect_annotated.png")
    cv2.imwrite(out_path, result.annotated_image)
    print(f"\n[SAVED] Visual confirmation image -> {out_path}")
    return True


def _prompt_psm() -> int:
    """Prompts for a PSM mode and re-prompts on invalid input instead of
    crashing with a ValueError on a bad int() cast."""
    while True:
        psm_in = input(f"PSM mode {sorted(PSM_MODES)} [default {DEFAULT_PSM}]: ").strip()
        if not psm_in:
            return DEFAULT_PSM
        try:
            psm = int(psm_in)
        except ValueError:
            print(f"  '{psm_in}' is not a number. Try one of {sorted(PSM_MODES)}.")
            continue
        if psm not in PSM_MODES:
            print(f"  {psm} is not a supported PSM mode. Try one of {sorted(PSM_MODES)}.")
            continue
        return psm


def interactive_menu() -> int:
    """Returns a process exit code (0 = success, 1 = at least one
    handled failure occurred)."""
    _banner()
    print("\nChoose your execution path:\n")
    print("  1) Path 1 - OCR (extract text from documents / invoices / headers)")
    print("  2) Path 2 - Object Detection (locate & label real-world objects)")
    print("  3) Run both paths on their respective sample images")
    print("  0) Exit")

    choice = input("\nEnter choice [1/2/3/0]: ").strip()
    ok = True

    if choice == "1":
        print("\nSample OCR images available:")
        print("  a) sample_images/sample_invoice.png      (recommended --psm 11)")
        print("  b) sample_images/sample_book_page.png    (recommended --psm 6)")
        print("  c) sample_images/sample_header_line.png  (recommended --psm 7)")
        img_choice = input("Pick a sample [a/b/c] or paste a custom path: ").strip()
        mapping = {
            "a": ("sample_images/sample_invoice.png", 11),
            "b": ("sample_images/sample_book_page.png", 6),
            "c": ("sample_images/sample_header_line.png", 7),
        }
        if img_choice in mapping:
            path, psm = mapping[img_choice]
        else:
            path = img_choice
            psm = _prompt_psm()
        ok = run_ocr(path, psm, lang="eng", no_deskew=False)

    elif choice == "2":
        print("\nSample object-detection images available:")
        print("  a) sample_images/sample_object_dog_bike_car.jpg")
        print("  b) sample_images/sample_object_person.jpg")
        print("  c) sample_images/sample_object_horses.jpg")
        img_choice = input("Pick a sample [a/b/c] or paste a custom path: ").strip()
        mapping = {
            "a": "sample_images/sample_object_dog_bike_car.jpg",
            "b": "sample_images/sample_object_person.jpg",
            "c": "sample_images/sample_object_horses.jpg",
        }
        path = mapping.get(img_choice, img_choice)
        ok = run_detection(path)

    elif choice == "3":
        ok_ocr = run_ocr("sample_images/sample_invoice.png", 11, "eng", False)
        ok_detect = run_detection("sample_images/sample_object_dog_bike_car.jpg")
        ok = ok_ocr and ok_detect

    elif choice == "0":
        print("Exiting. Keep innovating, keep learning, keep building the future.")
        return 0

    else:
        print(f"'{choice}' is not a valid option. Please choose 1, 2, 3, or 0.")
        return 1

    return 0 if ok else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project 4: Building the Machine's Optic Nerve — "
                    "Image & Text Recognition Pipeline",
    )
    parser.add_argument(
        "--path", choices=["ocr", "detect", "both"],
        help="Which recognition path to run. Omit to launch the interactive menu.",
    )
    parser.add_argument("--image", help="Path to the input image.")
    parser.add_argument(
        "--psm", type=int, default=DEFAULT_PSM, choices=sorted(PSM_MODES),
        help="Tesseract Page Segmentation Mode (OCR path only). "
             "3=auto, 6=block, 7=single line, 11=sparse.",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language pack (OCR path only).")
    parser.add_argument(
        "--no-deskew", action="store_true",
        help="Disable the deskew pre-processing step (OCR path only).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.path:
        sys.exit(interactive_menu())

    _banner()
    ok = True

    if args.path == "ocr":
        if not args.image:
            parser.error("--image is required for --path ocr")
        ok = run_ocr(args.image, args.psm, args.lang, args.no_deskew)

    elif args.path == "detect":
        if not args.image:
            parser.error("--image is required for --path detect")
        ok = run_detection(args.image)

    elif args.path == "both":
        if not args.image:
            parser.error("--image is required for --path both")
        ok_ocr = run_ocr(args.image, args.psm, args.lang, args.no_deskew)
        ok_detect = run_detection(args.image)
        ok = ok_ocr and ok_detect

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting cleanly.")
        sys.exit(130)
