"""
config.py
----------
Centralized configuration constants for Project 4, so every module
(and the README) references the same single source of truth.
"""

import os

# --- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

MOBILENET_PROTOTXT = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MOBILENET_WEIGHTS = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

# --- The 80% Gate ------------------------------------------------------------
# "The 80% Threshold: The Confidence Filter" — the absolute minimum
# standard for Project 4. Any detection below this is dropped.
CONFIDENCE_THRESHOLD = 0.80

# --- Path 1: OCR ---------------------------------------------------------
DEFAULT_TESSERACT_LANG = "eng"
DEFAULT_PSM = 3  # fully automatic page segmentation

# --- Path 2: Object Detection ----------------------------------------------
MOBILENET_INPUT_SIZE = (300, 300)
MOBILENET_MEAN_SUBTRACTION = 127.5
MOBILENET_SCALE_FACTOR = 1.0 / 127.5
