"""
core/confidence_gate.py
------------------------
"Decoding the Machine's Mind: Softmax & Confidence"

AI does not 'know' what an object or character is — it calculates the
statistical probability of what it might be (softmax output). Every
bounding box or text string produced by a model therefore carries a
confidence score, which is the model's own assessment of its accuracy.

"The 80% Threshold: The Confidence Filter"

Without a filter, an AI treats every guess with equal certainty, leading
to confident hallucinations and false positives. This module implements
the exact IF-statement gate shown in the Project 4 briefing:

    if confidence >= 0.80:
        draw_box_and_label()
    else:
        drop_detection()

High thresholds minimize False Positives but increase the risk of False
Negatives. In Project 4, 80% is the absolute minimum standard.
"""

from __future__ import annotations

from dataclasses import dataclass

CONFIDENCE_THRESHOLD: float = 0.80  # The 80% Gate — absolute minimum standard


@dataclass
class Detection:
    """A single recognized unit (a word/string for OCR, or an object for
    the detector) carrying its own confidence score."""

    label: str
    confidence: float          # 0.0 - 1.0
    box: tuple[int, int, int, int] | None = None  # (x, y, w, h) if applicable

    @property
    def confidence_pct(self) -> float:
        return round(self.confidence * 100, 2)

    @property
    def passes_gate(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD


def apply_confidence_gate(detections: list[Detection]) -> tuple[list[Detection], list[Detection]]:
    """Runs every detection through the 80% Gate.

        if confidence >= 0.80:  -> kept   (draw_box_and_label)
        else:                   -> dropped (drop_detection)

    Returns a tuple of (kept, dropped) so the pipeline can report both
    the accepted results AND how many low-confidence guesses were
    filtered out (useful for the Accuracy Benchmarking gatekeeper rule).
    """
    kept: list[Detection] = []
    dropped: list[Detection] = []

    for det in detections:
        if det.passes_gate:
            kept.append(det)
        else:
            dropped.append(det)

    return kept, dropped


def best_confidence(detections: list[Detection]) -> float:
    """Returns the single highest confidence score in a batch (0.0 if
    the batch is empty). Used for the 'minimum validated confidence
    score of 80% on the final output' gatekeeper check."""
    if not detections:
        return 0.0
    return max(d.confidence for d in detections)
