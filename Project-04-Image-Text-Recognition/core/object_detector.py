"""
core/object_detector.py
-------------------------
"Path 2: Object Detection with MobileNet-SSD"

    Core Objective:   Identifying & locating physical entities
    Primary Library:  cv2.dnn & MobileNet-SSD
    Pre-Processing:   4D Blob construction (blobFromImage)
    The Output:       (X, Y, W, H) Bounding Box Coordinates

The Backbone: MobileNet v3
    - Utilizes depthwise separable convolutions to filter input channels
      separately.
    - Optimized for high-speed, real-time inference on edge devices with
      minimal compute requirements.

Step 1: Blob Construction
    - We use cv2.dnn.blobFromImage.
    - Performs mean subtraction.
    - Scales the image to the required 300x300 network dimensions.

Decoding the Matrix: Anatomy of a Bounding Box
    - The network doesn't output an image; it outputs normalized spatial
      coordinates.
    - Coordinate Scaling: we multiply the normalized (X, Y, W, H) by the
      actual pixel width/height of the original image to physically draw
      the bounding box overlay.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from core.confidence_gate import Detection, apply_confidence_gate

# The 20 object classes MobileNet-SSD was trained on (Pascal VOC),
# index 0 is reserved for "background".
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

MODEL_INPUT_SIZE = (300, 300)      # required network dimensions
MEAN_SUBTRACTION = 127.5           # mean subtraction constant for MobileNet-SSD
SCALE_FACTOR = 1.0 / 127.5

_DEFAULT_PROTOTXT = os.path.join("models", "MobileNetSSD_deploy.prototxt")
_DEFAULT_WEIGHTS = os.path.join("models", "MobileNetSSD_deploy.caffemodel")


class DetectionResult:
    """The complete, inspectable result of running the Object Detection
    pipeline."""

    def __init__(self) -> None:
        self.all_detections: list[Detection] = []
        self.kept_detections: list[Detection] = []
        self.dropped_detections: list[Detection] = []
        self.annotated_image: np.ndarray | None = None
        self.blob_shape: tuple[int, ...] | None = None

    @property
    def best_confidence(self) -> float:
        if not self.all_detections:
            return 0.0
        return max(d.confidence for d in self.all_detections)

    @property
    def gatekeeper_passed(self) -> bool:
        return len(self.kept_detections) > 0


_net_cache: cv2.dnn_Net | None = None


def load_model(
    prototxt_path: str = _DEFAULT_PROTOTXT,
    weights_path: str = _DEFAULT_WEIGHTS,
) -> cv2.dnn_Net:
    """Loads (and caches) the pre-trained MobileNet-SSD Caffe model via
    Transfer Learning — we inherit millions of ImageNet-derived visual
    concepts instead of training from scratch."""
    global _net_cache
    if _net_cache is not None:
        return _net_cache

    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            "MobileNet-SSD model files not found. Expected:\n"
            f"  {prototxt_path}\n  {weights_path}\n"
            "Run 'python models/download_models.py' first."
        )

    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    _net_cache = net
    return net


def detect_objects(
    image: np.ndarray,
    net: cv2.dnn_Net | None = None,
) -> DetectionResult:
    """Runs the full Object Detection pipeline: blob construction ->
    forward pass -> coordinate scaling -> 80% gate.

    Args:
        image: BGR image loaded via cv2.imread.
        net:   an already-loaded cv2.dnn_Net (optional; auto-loads
               the default MobileNet-SSD model if omitted).
    """
    if net is None:
        net = load_model()

    result = DetectionResult()
    (h, w) = image.shape[:2]

    # --- Step 1: Blob Construction ---------------------------------------
    # cv2.dnn.blobFromImage performs mean subtraction and scales the
    # image to the required 300x300 network dimensions, producing the
    # 4D blob (N, C, H, W) the network expects.
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=SCALE_FACTOR,
        size=MODEL_INPUT_SIZE,
        mean=(MEAN_SUBTRACTION, MEAN_SUBTRACTION, MEAN_SUBTRACTION),
        swapRB=False,
        crop=False,
    )
    result.blob_shape = blob.shape

    net.setInput(blob)
    detections = net.forward()  # shape: (1, 1, N, 7)

    parsed: list[Detection] = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        label = VOC_CLASSES[class_id] if 0 <= class_id < len(VOC_CLASSES) else "unknown"

        if label == "background":
            continue

        # --- Coordinate Scaling -------------------------------------------
        # The network outputs normalized [0, 1] coordinates. We multiply
        # by the *actual* pixel width/height of the original image to
        # physically place the bounding box.
        box_norm = detections[0, 0, i, 3:7]
        (start_x, start_y, end_x, end_y) = (box_norm * np.array([w, h, w, h])).astype(int)
        start_x, start_y = max(0, start_x), max(0, start_y)
        end_x, end_y = min(w - 1, end_x), min(h - 1, end_y)
        box_w, box_h = max(0, end_x - start_x), max(0, end_y - start_y)

        parsed.append(
            Detection(
                label=label,
                confidence=confidence,
                box=(int(start_x), int(start_y), int(box_w), int(box_h)),
            )
        )

    result.all_detections = parsed

    # --- The 80% Confidence Gate ------------------------------------------
    kept, dropped = apply_confidence_gate(parsed)
    result.kept_detections = kept
    result.dropped_detections = dropped

    # --- Visual Confirmation: draw labeled bounding boxes ------------------
    annotated = image.copy()
    for det in kept:
        x, y, bw, bh = det.box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        label_text = f"{det.label}: {det.confidence_pct}%"
        text_y = y - 10 if y - 10 > 10 else y + 15
        cv2.putText(
            annotated, label_text, (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA,
        )

    result.annotated_image = annotated
    return result
