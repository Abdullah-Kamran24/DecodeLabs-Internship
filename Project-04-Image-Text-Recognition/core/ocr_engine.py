"""
core/ocr_engine.py
-------------------
"Path 1: Optical Character Recognition (OCR)"

The Engine: pytesseract is our Python wrapper for Google's Tesseract
engine, which uses a convolutional + bi-directional LSTM pipeline to
read sequences of characters.

    Core Objective:   Extracting machine-readable strings
    Primary Library:  pytesseract
    Pre-Processing:   Grayscale, blur, adaptive thresholding
    The Output:       Formatted text strings (with per-word confidence)

Tuning the PSM (Page Segmentation Mode) — layout configuration is
critical for accuracy:

    --psm 3  : Fully automatic page segmentation (default, varied layouts)
    --psm 6  : Assume a single uniform block of text (book pages)
    --psm 7  : Treat the image as a single text line (number plates / headers)
    --psm 11 : Sparse text — find as much text as possible, no particular order
               (scattered text such as invoices)
"""

from __future__ import annotations

import os
import platform
import shutil

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from core.confidence_gate import Detection, apply_confidence_gate
from core.preprocessing import run_full_pipeline, PreprocessingReport

# Human-friendly names for each supported Page Segmentation Mode, mapped
# exactly to "The Perception Matrix" / "Tuning the PSM" table.
PSM_MODES: dict[int, str] = {
    3: "Fully automatic page segmentation (default, varied layouts)",
    6: "Single uniform block of text (book pages)",
    7: "Single text line (number plates / headers)",
    11: "Sparse, scattered text (invoices)",
}

DEFAULT_PSM = 3

# The Tesseract *engine binary* (not the pip package) is a separate,
# OS-level install. pytesseract is only a thin wrapper around it. If the
# binary isn't already on PATH, we try the well-known default install
# locations for each OS before giving up, so users don't hit a raw
# traceback just because PATH wasn't updated.
_WINDOWS_DEFAULT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]
_MACOS_DEFAULT_PATHS = [
    "/opt/homebrew/bin/tesseract",   # Apple Silicon Homebrew
    "/usr/local/bin/tesseract",       # Intel Homebrew
]
_LINUX_DEFAULT_PATHS = [
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]


class TesseractNotConfiguredError(RuntimeError):
    """Raised when the Tesseract OCR engine binary cannot be located
    anywhere on the system, with a clear, actionable fix for the user."""


def _locate_tesseract_binary() -> str | None:
    """Best-effort auto-detection of the Tesseract engine binary.

    Order of precedence:
        1. Whatever pytesseract is already configured to use, if valid.
        2. Whatever 'tesseract' resolves to on PATH (shutil.which).
        3. The well-known default install directory for the current OS.
    """
    current_cmd = pytesseract.pytesseract.tesseract_cmd
    if current_cmd and os.path.isfile(current_cmd):
        return current_cmd

    which_result = shutil.which("tesseract")
    if which_result:
        return which_result

    system = platform.system()
    if system == "Windows":
        candidates = _WINDOWS_DEFAULT_PATHS
    elif system == "Darwin":
        candidates = _MACOS_DEFAULT_PATHS
    else:
        candidates = _LINUX_DEFAULT_PATHS

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def ensure_tesseract_available() -> str:
    """Locates the Tesseract engine binary, wires it into pytesseract,
    and returns the resolved path. Raises TesseractNotConfiguredError
    with clear, OS-specific setup instructions if it cannot be found
    anywhere, instead of letting a raw FileNotFoundError/traceback
    bubble up from deep inside pytesseract/subprocess.
    """
    resolved = _locate_tesseract_binary()

    if resolved is None:
        system = platform.system()
        if system == "Windows":
            fix = (
                "Tesseract OCR is not installed, or it is installed but not "
                "on your PATH.\n\n"
                "  1. Download & run the installer:\n"
                "     https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  2. During install, note the install folder (default:\n"
                r"     C:\Program Files\Tesseract-OCR)" "\n"
                "  3. Either add that folder to your system PATH and open a\n"
                "     NEW terminal, OR set it directly in code:\n\n"
                "         import pytesseract\n"
                r"         pytesseract.pytesseract.tesseract_cmd = "
                r'r"C:\Program Files\Tesseract-OCR\tesseract.exe"' "\n\n"
                "  4. Verify with:  tesseract --version"
            )
        elif system == "Darwin":
            fix = (
                "Tesseract OCR is not installed, or not on your PATH.\n\n"
                "  1. Install via Homebrew:   brew install tesseract\n"
                "  2. Verify with:            tesseract --version"
            )
        else:
            fix = (
                "Tesseract OCR is not installed, or not on your PATH.\n\n"
                "  1. Install it:  sudo apt-get update && "
                "sudo apt-get install -y tesseract-ocr\n"
                "  2. Verify with: tesseract --version"
            )
        raise TesseractNotConfiguredError(
            "\n\n[Path 1 / OCR] Cannot find the Tesseract OCR engine binary.\n"
            "pytesseract is only a Python wrapper — it requires the actual\n"
            "Tesseract program to be installed separately at the OS level.\n\n"
            f"{fix}\n"
        )

    pytesseract.pytesseract.tesseract_cmd = resolved
    return resolved


class OCRResult:
    """The complete, inspectable result of running the OCR pipeline."""

    def __init__(self) -> None:
        self.psm_used: int = DEFAULT_PSM
        self.preprocessing_report: PreprocessingReport | None = None
        self.all_detections: list[Detection] = []
        self.kept_detections: list[Detection] = []
        self.dropped_detections: list[Detection] = []
        self.full_text: str = ""
        self.binary_image: np.ndarray | None = None
        self.annotated_image: np.ndarray | None = None

    @property
    def best_confidence(self) -> float:
        if not self.all_detections:
            return 0.0
        return max(d.confidence for d in self.all_detections)

    @property
    def gatekeeper_passed(self) -> bool:
        """Milestone 3 / 4 check: at least one legible string with a
        confidence score >= the 80% minimum standard."""
        return len(self.kept_detections) > 0


def _validate_psm(psm: int) -> int:
    if psm not in PSM_MODES:
        raise ValueError(
            f"Unsupported PSM mode: {psm}. Choose one of {sorted(PSM_MODES)}."
        )
    return psm


def recognize_text(
    image: np.ndarray,
    psm: int = DEFAULT_PSM,
    lang: str = "eng",
    apply_deskew: bool = True,
) -> OCRResult:
    """Runs the full OCR pipeline: pre-processing -> Tesseract -> 80% gate.

    Args:
        image:       BGR image loaded via cv2.imread.
        psm:         Tesseract Page Segmentation Mode (3, 6, 7, or 11).
        lang:        Tesseract language pack (default English).
        apply_deskew: whether to run the deskew correction step.

    Returns:
        An OCRResult carrying the recognized text, per-word confidences,
        the pre-processing report, and an annotated visual output.
    """
    psm = _validate_psm(psm)

    # Resolve + wire up the Tesseract engine binary *before* doing any
    # image work, so failures surface with a clear, actionable message
    # instead of a raw traceback from deep inside pytesseract/subprocess.
    ensure_tesseract_available()

    result = OCRResult()
    result.psm_used = psm

    # --- Steps 1-4: The Logic Skeleton pre-processing pipeline ---------
    processed_gray, binary_image, report = run_full_pipeline(
        image, apply_deskew=apply_deskew
    )
    result.preprocessing_report = report
    result.binary_image = binary_image

    # --- Tesseract configuration -----------------------------------------
    tess_config = f"--oem 3 --psm {psm}"

    # image_to_data gives us per-word bounding boxes + confidence, which
    # is required both for the visual confirmation and the 80% gate.
    data = pytesseract.image_to_data(
        binary_image,
        lang=lang,
        config=tess_config,
        output_type=Output.DICT,
    )

    detections: list[Detection] = []
    words: list[str] = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0

        if not text or conf < 0:
            continue  # Tesseract emits -1 confidence for non-text regions

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        detections.append(
            Detection(label=text, confidence=conf / 100.0, box=(x, y, w, h))
        )
        words.append(text)

    result.all_detections = detections
    result.full_text = " ".join(words)

    # --- The 80% Confidence Gate ------------------------------------------
    kept, dropped = apply_confidence_gate(detections)
    result.kept_detections = kept
    result.dropped_detections = dropped

    # --- Visual Confirmation: draw green boxes for kept, red for dropped --
    annotated = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    for det in kept:
        x, y, w, h = det.box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{det.label} ({det.confidence_pct}%)"
        cv2.putText(
            annotated, label, (x, max(y - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
        )
    for det in dropped:
        x, y, w, h = det.box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 220), 1)

    result.annotated_image = annotated
    return result
