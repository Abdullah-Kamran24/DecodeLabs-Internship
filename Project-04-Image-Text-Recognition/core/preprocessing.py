"""
core/preprocessing.py
----------------------
"The Logic Skeleton: Systematic Image Pre-Processing"

Raw visual data is cluttered with shadows, chromatic noise, and uneven
lighting. Before any recognition model (OCR or object detector) can trust
the pixels it is given, the image must be pushed through a deterministic
cleanup pipeline:

    Step 1: Grayscale Conversion  -> collapses the 3D RGB matrix into a
                                      1D intensity matrix (removes color).
    Step 2: Gaussian Blur         -> smooths micro-imperfections / noise.
    Step 3: Deskewing             -> rotates tilted text back to a
                                      horizontal baseline.
    Step 4: Adaptive Thresholding -> forces every pixel to commit to
                                      black or white (Otsu's method).

This module is intentionally framework-agnostic: both the OCR path
(pytesseract) and the Object Detection path (cv2.dnn / MobileNet-SSD)
import from here so the "Pre-Processing Integrity" gatekeeper rule is
satisfied identically on both paths.
"""

from __future__ import annotations

import cv2
import numpy as np


class PreprocessingReport:
    """Tiny data holder so the pipeline can prove *what* it did.

    The Gatekeeper Rule (Milestone 2: Pre-Processing Integrity) demands a
    *demonstrable* execution of grayscale conversion + adaptive
    thresholding, not just a black-box result. This object is what the
    validator inspects.
    """

    def __init__(self) -> None:
        self.grayscale_applied: bool = False
        self.blur_applied: bool = False
        self.deskew_applied: bool = False
        self.deskew_angle_deg: float = 0.0
        self.threshold_applied: bool = False
        self.otsu_cutoff: float | None = None

    def as_dict(self) -> dict:
        return {
            "grayscale_applied": self.grayscale_applied,
            "blur_applied": self.blur_applied,
            "deskew_applied": self.deskew_applied,
            "deskew_angle_deg": round(self.deskew_angle_deg, 3),
            "threshold_applied": self.threshold_applied,
            "otsu_cutoff": self.otsu_cutoff,
        }


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Step 1: Grayscale Conversion.

    Collapses the 3-channel (R, G, B) matrix into a single intensity
    channel. This removes distracting color data that recognition models
    do not need to read shapes or text strokes.
    """
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Step 2: Gaussian Blur.

    Smooths the image to eliminate micro-imperfections and sensor/artifact
    noise before edges and characters are analyzed.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # OpenCV requires an odd kernel size
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def deskew(gray_image: np.ndarray) -> tuple[np.ndarray, float]:
    """Step 3: Deskewing.

    Calculates the rotation angle of the dominant text/content block and
    rotates the image so the baseline is perfectly horizontal. Returns the
    corrected image and the angle (in degrees) that was applied.
    """
    # Invert + threshold so text/foreground pixels are white on black,
    # which is what cv2.minAreaRect expects to find contour points.
    inverted = cv2.bitwise_not(gray_image)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        # Not enough foreground pixels to reliably estimate a skew angle
        # (e.g. a near-blank image) -> skip deskewing safely.
        return gray_image, 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns an angle in [-90, 0); normalize it to the
    # small correction we actually want to apply.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def adaptive_threshold(gray_image: np.ndarray) -> tuple[np.ndarray, float]:
    """Step 4: Adaptive Thresholding — Forcing the Binary Decision.

    Otsu's method automatically calculates the optimal global cutoff
    intensity and applies:

        if pixel_intensity >= cutoff: pixel = 255 (white)
        if pixel_intensity <  cutoff: pixel = 0   (black)

    Returns the binarized image and the calculated cutoff value.
    """
    cutoff, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, cutoff


def run_full_pipeline(
    image: np.ndarray,
    blur_kernel: int = 5,
    apply_deskew: bool = True,
) -> tuple[np.ndarray, np.ndarray, PreprocessingReport]:
    """Runs the complete Logic Skeleton pipeline end-to-end.

    Returns:
        gray_for_ocr:  grayscale + blurred + deskewed image (best fed to
                       pytesseract, which does its own internal binarization).
        binary_image:  the fully thresholded black/white image (useful for
                       visual proof / contour-based work).
        report:        a PreprocessingReport documenting each step taken.
    """
    report = PreprocessingReport()

    gray = to_grayscale(image)
    report.grayscale_applied = True

    blurred = gaussian_blur(gray, blur_kernel)
    report.blur_applied = True

    if apply_deskew:
        deskewed, angle = deskew(blurred)
        report.deskew_applied = True
        report.deskew_angle_deg = angle
    else:
        deskewed = blurred

    binary_image, cutoff = adaptive_threshold(deskewed)
    report.threshold_applied = True
    report.otsu_cutoff = float(cutoff)

    return deskewed, binary_image, report
