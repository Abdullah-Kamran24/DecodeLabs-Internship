"""
tests/test_pipeline.py
------------------------
Lightweight sanity tests for Project 4's core modules. These do not
require pytest — run directly:

    python tests/test_pipeline.py

or with pytest if installed:

    pytest tests/
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessing import (
    to_grayscale, gaussian_blur, deskew, adaptive_threshold, run_full_pipeline,
)
from core.confidence_gate import Detection, apply_confidence_gate, CONFIDENCE_THRESHOLD
from core.ocr_engine import recognize_text
from core.object_detector import detect_objects, load_model

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, condition: bool) -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}")


def test_grayscale_collapses_channels():
    color = np.zeros((50, 50, 3), dtype=np.uint8)
    color[:, :, 0] = 10
    color[:, :, 1] = 20
    color[:, :, 2] = 30
    gray = to_grayscale(color)
    check("grayscale output has 2 dimensions", gray.ndim == 2)
    check("grayscale shape matches input HxW", gray.shape == (50, 50))


def test_blur_reduces_noise_variance():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    blurred = gaussian_blur(noisy, kernel_size=7)
    check("blur reduces pixel variance", float(np.var(blurred)) < float(np.var(noisy)))


def test_threshold_is_strictly_binary():
    gray = np.linspace(0, 255, 100).astype(np.uint8).reshape(10, 10)
    binary, cutoff = adaptive_threshold(gray)
    unique_vals = set(np.unique(binary).tolist())
    check("thresholded image only contains 0 and 255", unique_vals.issubset({0, 255}))
    check("Otsu cutoff is within valid pixel range", 0 <= cutoff <= 255)


def test_confidence_gate_80_percent():
    detections = [
        Detection("high_conf", 0.95),
        Detection("borderline", 0.80),
        Detection("just_below", 0.79),
        Detection("low_conf", 0.30),
    ]
    kept, dropped = apply_confidence_gate(detections)
    check("gate keeps exactly the >= 80% detections", len(kept) == 2)
    check("gate drops exactly the < 80% detections", len(dropped) == 2)
    check("threshold constant is 0.80", CONFIDENCE_THRESHOLD == 0.80)


def test_full_ocr_pipeline_on_sample():
    img_path = "sample_images/sample_invoice.png"
    if not os.path.exists(img_path):
        print(f"  [SKIP] sample image missing: {img_path}")
        return
    image = cv2.imread(img_path)
    result = recognize_text(image, psm=11)
    check("OCR extracts at least one word", len(result.all_detections) > 0)
    check("OCR gatekeeper passes 80% gate", result.gatekeeper_passed)
    check("OCR preprocessing report shows grayscale+threshold applied",
          result.preprocessing_report.grayscale_applied and
          result.preprocessing_report.threshold_applied)


def test_full_detection_pipeline_on_sample():
    img_path = "sample_images/sample_object_dog_bike_car.jpg"
    proto = "models/MobileNetSSD_deploy.prototxt"
    weights = "models/MobileNetSSD_deploy.caffemodel"
    if not (os.path.exists(img_path) and os.path.exists(proto) and os.path.exists(weights)):
        print("  [SKIP] detection sample or model files missing")
        return
    net = load_model(proto, weights)
    image = cv2.imread(img_path)
    result = detect_objects(image, net)
    check("Detector produces a 4D blob (1,3,300,300)", result.blob_shape == (1, 3, 300, 300))
    check("Detector finds at least one object above 80%", result.gatekeeper_passed)
    labels = {d.label for d in result.kept_detections}
    check("Detector correctly identifies 'dog'", "dog" in labels)


def main() -> None:
    print("Running Project 4 sanity tests...\n")
    print("core.preprocessing:")
    test_grayscale_collapses_channels()
    test_blur_reduces_noise_variance()
    test_threshold_is_strictly_binary()

    print("\ncore.confidence_gate:")
    test_confidence_gate_80_percent()

    print("\nend-to-end OCR pipeline:")
    test_full_ocr_pipeline_on_sample()

    print("\nend-to-end Object Detection pipeline:")
    test_full_detection_pipeline_on_sample()

    print(f"\n{'=' * 50}\nResults: {PASS_COUNT} passed, {FAIL_COUNT} failed\n{'=' * 50}")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
