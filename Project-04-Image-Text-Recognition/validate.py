#!/usr/bin/env python3
"""
validate.py
------------
"The Gatekeeper Rule: Milestone Validation"

The Standard: To complete Project 4, your script must pass four
uncompromising technical validations:

    1. Library Integration      -> Seamless, error-free implementation
                                    of pytesseract or cv2.dnn.
    2. Pre-Processing Integrity  -> Demonstrable execution of Grayscale
                                    conversion and Adaptive Thresholding
                                    to separate foreground from noise.
    3. Accuracy Benchmarking     -> A minimum validated confidence score
                                    of 80% on the final output.
    4. Visual Confirmation       -> Generation of a pristine visual output
                                    (legible OCR string or accurate
                                    bounding boxes with labels).

Run this after setting up the project to confirm every milestone is met
end-to-end, on both paths, using the bundled sample images:

    python validate.py
"""

from __future__ import annotations

import os
import sys

import cv2

from core.ocr_engine import recognize_text
from core.object_detector import detect_objects, load_model

PASS = "\u2705 PASS"
FAIL = "\u274c FAIL"

SAMPLE_OCR_IMAGE = "sample_images/sample_invoice.png"
SAMPLE_OCR_PSM = 11
SAMPLE_DETECT_IMAGE = "sample_images/sample_object_dog_bike_car.jpg"


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f" {title}")
    print("=" * 72)


def validate_ocr_path() -> bool:
    _header("VALIDATING PATH 1 — OCR (pytesseract)")
    all_ok = True

    # Milestone 1: Library Integration
    try:
        image = cv2.imread(SAMPLE_OCR_IMAGE)
        if image is None:
            raise FileNotFoundError(SAMPLE_OCR_IMAGE)
        result = recognize_text(image, psm=SAMPLE_OCR_PSM)
        print(f"[1] Library Integration (pytesseract)....... {PASS}")
    except Exception as exc:  # noqa: BLE001
        print(f"[1] Library Integration (pytesseract)....... {FAIL}  ({exc})")
        return False

    # Milestone 2: Pre-Processing Integrity
    report = result.preprocessing_report.as_dict()
    m2_ok = report["grayscale_applied"] and report["threshold_applied"]
    print(f"[2] Pre-Processing Integrity................. {PASS if m2_ok else FAIL}")
    print(f"      grayscale_applied = {report['grayscale_applied']}, "
          f"threshold_applied = {report['threshold_applied']}, "
          f"otsu_cutoff = {report['otsu_cutoff']}")
    all_ok &= m2_ok

    # Milestone 3: Accuracy Benchmarking (>= 80%)
    m3_ok = result.best_confidence >= 0.80
    print(f"[3] Accuracy Benchmarking (>= 80%)............ {PASS if m3_ok else FAIL}")
    print(f"      best_confidence = {result.best_confidence * 100:.2f}%")
    all_ok &= m3_ok

    # Milestone 4: Visual Confirmation
    out_path = "output/_validate_ocr_annotated.png"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(out_path, result.annotated_image)
    m4_ok = os.path.exists(out_path) and len(result.full_text.strip()) > 0
    print(f"[4] Visual Confirmation (legible output)...... {PASS if m4_ok else FAIL}")
    print(f"      saved -> {out_path}")
    all_ok &= m4_ok

    return bool(all_ok)


def validate_detection_path() -> bool:
    _header("VALIDATING PATH 2 — Object Detection (MobileNet-SSD)")
    all_ok = True

    # Milestone 1: Library Integration
    try:
        net = load_model()
        image = cv2.imread(SAMPLE_DETECT_IMAGE)
        if image is None:
            raise FileNotFoundError(SAMPLE_DETECT_IMAGE)
        result = detect_objects(image, net)
        print(f"[1] Library Integration (cv2.dnn)............ {PASS}")
    except Exception as exc:  # noqa: BLE001
        print(f"[1] Library Integration (cv2.dnn)............ {FAIL}  ({exc})")
        return False

    # Milestone 2: Pre-Processing Integrity (blob construction)
    m2_ok = result.blob_shape == (1, 3, 300, 300)
    print(f"[2] Pre-Processing Integrity (blob 300x300)... {PASS if m2_ok else FAIL}")
    print(f"      blob_shape = {result.blob_shape}")
    all_ok &= m2_ok

    # Milestone 3: Accuracy Benchmarking (>= 80%)
    m3_ok = result.best_confidence >= 0.80
    print(f"[3] Accuracy Benchmarking (>= 80%)............ {PASS if m3_ok else FAIL}")
    print(f"      best_confidence = {result.best_confidence * 100:.2f}%")
    all_ok &= m3_ok

    # Milestone 4: Visual Confirmation
    out_path = "output/_validate_detect_annotated.png"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(out_path, result.annotated_image)
    m4_ok = os.path.exists(out_path) and len(result.kept_detections) > 0
    print(f"[4] Visual Confirmation (labeled boxes)....... {PASS if m4_ok else FAIL}")
    print(f"      saved -> {out_path}")
    all_ok &= m4_ok

    return bool(all_ok)


def main() -> None:
    print("PROJECT 4 GATEKEEPER VALIDATION")
    print("Architecting Machine Autonomy — confirming all four milestones "
          "on both recognition paths.")

    ocr_ok = validate_ocr_path()
    detect_ok = validate_detection_path()

    _header("FINAL RESULT")
    print(f"  Path 1 (OCR)              : {PASS if ocr_ok else FAIL}")
    print(f"  Path 2 (Object Detection) : {PASS if detect_ok else FAIL}")

    if ocr_ok and detect_ok:
        print("\nAll Gatekeeper milestones passed. Project 4 requirements met.")
        sys.exit(0)
    else:
        print("\nOne or more milestones failed. Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
