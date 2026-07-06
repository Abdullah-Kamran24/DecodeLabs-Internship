#!/usr/bin/env python3
"""
check_environment.py
-----------------------
A one-shot preflight diagnostic. Run this FIRST, before main.py, on any
new machine (especially Windows) to catch setup problems — like a
missing Tesseract OCR binary — with a clear fix, instead of discovering
them mid-run as a traceback.

Usage:
    python check_environment.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys

OK = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"

ROOT = os.path.dirname(os.path.abspath(__file__))
results: list[tuple[bool, str]] = []  # (passed, message)


def check(label: str, passed: bool, detail: str = "") -> None:
    tag = OK if passed else FAIL
    line = f"{tag} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    results.append((passed, label))


def warn(label: str, detail: str = "") -> None:
    line = f"{WARN} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


def check_python_version() -> None:
    major, minor = sys.version_info.major, sys.version_info.minor
    passed = (major, minor) >= (3, 9)
    check("Python >= 3.9", passed, f"found {platform.python_version()}")


def check_module(module_name: str, friendly_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        check(f"Python package '{friendly_name}' importable", True)
        return True
    except ImportError as exc:
        check(f"Python package '{friendly_name}' importable", False, str(exc))
        return False


def check_tesseract_binary() -> None:
    """Checks that the Tesseract OCR *engine* (not just the pip wrapper)
    is reachable, using the same auto-detection logic main.py relies on."""
    try:
        sys.path.insert(0, ROOT)
        from core.ocr_engine import ensure_tesseract_available, TesseractNotConfiguredError
        try:
            resolved_path = ensure_tesseract_available()
            check("Tesseract OCR engine binary found", True, resolved_path)
        except TesseractNotConfiguredError as exc:
            check("Tesseract OCR engine binary found", False)
            print(str(exc))
    except Exception as exc:  # noqa: BLE001
        check("Tesseract OCR engine binary found", False, f"could not run check: {exc}")


def check_model_files() -> None:
    prototxt = os.path.join(ROOT, "models", "MobileNetSSD_deploy.prototxt")
    weights = os.path.join(ROOT, "models", "MobileNetSSD_deploy.caffemodel")

    proto_ok = os.path.isfile(prototxt) and os.path.getsize(prototxt) > 10_000
    check("MobileNet-SSD prototxt present", proto_ok, prototxt)

    weights_ok = os.path.isfile(weights) and os.path.getsize(weights) > 20_000_000
    check(
        "MobileNet-SSD caffemodel present (~23 MB)", weights_ok,
        f"{weights} "
        f"({os.path.getsize(weights):,} bytes)" if os.path.isfile(weights) else weights,
    )
    if not weights_ok:
        print("        Fix: run  python models/download_models.py")


def check_sample_images() -> None:
    expected = [
        "sample_images/sample_invoice.png",
        "sample_images/sample_book_page.png",
        "sample_images/sample_header_line.png",
        "sample_images/sample_object_dog_bike_car.jpg",
        "sample_images/sample_object_person.jpg",
        "sample_images/sample_object_horses.jpg",
    ]
    missing = [p for p in expected if not os.path.isfile(os.path.join(ROOT, p))]
    check("All bundled sample images present", len(missing) == 0,
          "missing: " + ", ".join(missing) if missing else "6/6 found")


def check_model_loads() -> None:
    try:
        sys.path.insert(0, ROOT)
        import cv2
        from core.object_detector import load_model
        os.chdir(ROOT)
        load_model()
        check("MobileNet-SSD model loads via cv2.dnn", True)
    except Exception as exc:  # noqa: BLE001
        check("MobileNet-SSD model loads via cv2.dnn", False, str(exc))


def main() -> None:
    print("=" * 72)
    print(" PROJECT 4 — ENVIRONMENT PREFLIGHT CHECK")
    print(f" Platform: {platform.system()} {platform.release()} | "
          f"Python {platform.python_version()}")
    print("=" * 72 + "\n")

    check_python_version()
    check_module("cv2", "opencv-python")
    check_module("numpy", "numpy")
    check_module("PIL", "Pillow")
    has_pytesseract = check_module("pytesseract", "pytesseract")
    if has_pytesseract:
        check_tesseract_binary()
    check_model_files()
    check_sample_images()
    check_model_loads()

    n_pass = sum(1 for p, _ in results if p)
    n_total = len(results)

    print("\n" + "=" * 72)
    print(f" RESULT: {n_pass}/{n_total} checks passed")
    print("=" * 72)

    if n_pass == n_total:
        print("\nEnvironment is fully ready. Run:  python main.py")
        sys.exit(0)
    else:
        print("\nSome checks failed — see [FAIL] lines above for the exact fix.")
        print("Re-run this script after applying the fixes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
