"""
sample_images/generate_samples.py
-----------------------------------
Generates three synthetic OCR test images so Path 1 can be exercised
against every Page Segmentation Mode described in "The Perception
Matrix":

    sample_invoice.png       -> --psm 11 (sparse, scattered text)
    sample_book_page.png     -> --psm 6  (single uniform block of text)
    sample_header_line.png   -> --psm 7  (single text line)

Each image is deliberately given mild real-world noise (Gaussian noise,
a slight rotation / skew, and uneven lighting via a soft gradient) so
the pre-processing pipeline (grayscale -> blur -> deskew -> adaptive
threshold) has real work to do, matching "The Problem: Raw visual data
is cluttered with shadows, chromatic noise, and uneven lighting."

Run once:  python sample_images/generate_samples.py
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"


def _add_noise_and_shadow(img: Image.Image, skew_deg: float = 0.0) -> Image.Image:
    """Adds a soft lighting gradient + Gaussian noise, then rotates the
    canvas slightly to simulate a real photographed/scanned document."""
    arr = np.array(img).astype(np.float32)

    # Uneven lighting: soft diagonal gradient shadow.
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 25 * (xx / w) - 12  # ranges roughly -12..+13
    for c in range(3):
        arr[:, :, c] += gradient

    # Chromatic / sensor noise.
    noise = np.random.normal(0, 6, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    noisy_img = Image.fromarray(arr)

    if skew_deg:
        noisy_img = noisy_img.rotate(
            skew_deg, resample=Image.BICUBIC, expand=True, fillcolor=(250, 248, 240)
        )
    return noisy_img


def make_invoice() -> None:
    """Sparse, scattered text laid out like a real invoice -> --psm 11."""
    w, h = 900, 1100
    img = Image.new("RGB", (w, h), (250, 248, 240))
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_BOLD, 34)
    label_font = ImageFont.truetype(FONT_REGULAR, 20)
    body_font = ImageFont.truetype(FONT_REGULAR, 18)

    draw.text((40, 40), "DECODELABS INVOICE", font=title_font, fill=(20, 20, 20))
    draw.line([(40, 95), (860, 95)], fill=(20, 20, 20), width=3)

    draw.text((40, 130), "Invoice #: INV-2026-00458", font=label_font, fill=(30, 30, 30))
    draw.text((40, 165), "Date: 2026-07-05", font=label_font, fill=(30, 30, 30))
    draw.text((600, 130), "Status: PAID", font=label_font, fill=(0, 110, 0))

    draw.text((40, 230), "Bill To:", font=label_font, fill=(30, 30, 30))
    draw.text((40, 260), "Architect Onboarding Program", font=body_font, fill=(30, 30, 30))
    draw.text((40, 285), "DecodeLabs Training Division", font=body_font, fill=(30, 30, 30))

    # Table header
    draw.rectangle([(40, 350), (860, 385)], fill=(230, 230, 225))
    draw.text((50, 358), "ITEM", font=label_font, fill=(0, 0, 0))
    draw.text((520, 358), "QTY", font=label_font, fill=(0, 0, 0))
    draw.text((620, 358), "PRICE", font=label_font, fill=(0, 0, 0))
    draw.text((740, 358), "TOTAL", font=label_font, fill=(0, 0, 0))

    rows = [
        ("Pytesseract OCR Engine License", "1", "$0.00", "$0.00"),
        ("OpenCV Preprocessing Toolkit", "1", "$0.00", "$0.00"),
        ("MobileNet-SSD Model Access", "1", "$0.00", "$0.00"),
        ("Certification Processing Fee", "1", "$49.00", "$49.00"),
    ]
    y = 400
    for item, qty, price, total in rows:
        draw.text((50, y), item, font=body_font, fill=(20, 20, 20))
        draw.text((520, y), qty, font=body_font, fill=(20, 20, 20))
        draw.text((620, y), price, font=body_font, fill=(20, 20, 20))
        draw.text((740, y), total, font=body_font, fill=(20, 20, 20))
        y += 40

    draw.line([(40, y + 10), (860, y + 10)], fill=(20, 20, 20), width=2)
    draw.text((620, y + 30), "GRAND TOTAL: $49.00", font=label_font, fill=(0, 0, 0))

    draw.text((40, 950), "Thank you for building the future with DecodeLabs.",
              font=body_font, fill=(60, 60, 60))
    draw.text((40, 1000), "Questions? support@decodelabs.example", font=body_font, fill=(60, 60, 60))

    final = _add_noise_and_shadow(img, skew_deg=-2.3)
    final.convert("RGB").save(os.path.join(OUT_DIR, "sample_invoice.png"))


def make_book_page() -> None:
    """A single uniform block of paragraph text -> --psm 6."""
    w, h = 900, 700
    img = Image.new("RGB", (w, h), (248, 246, 238))
    draw = ImageDraw.Draw(img)

    heading_font = ImageFont.truetype(FONT_SERIF, 30)
    body_font = ImageFont.truetype(FONT_SERIF, 20)

    draw.text((60, 50), "Chapter 4: The Optic Nerve", font=heading_font, fill=(15, 15, 15))

    paragraph = (
        "To a machine an image is not a picture it is a massive three "
        "dimensional array of numbers. Height and width describe the "
        "spatial pixel resolution while depth carries three color "
        "channels for red green and blue. Every single pixel channel "
        "holds an intensity value between zero and two hundred fifty "
        "five. Altering a single coordinate directly alters the "
        "machine's reality. This is the foundation upon which every "
        "recognition pipeline is built, from simple optical character "
        "recognition to deep convolutional object detectors."
    )

    # naive word wrap
    words = paragraph.split()
    lines, current = [], ""
    max_width = w - 120
    for word in words:
        trial = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), trial, font=body_font)
        if bbox[2] - bbox[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)

    y = 130
    for line in lines:
        draw.text((60, y), line, font=body_font, fill=(25, 25, 25))
        y += 34

    final = _add_noise_and_shadow(img, skew_deg=1.1)
    final.convert("RGB").save(os.path.join(OUT_DIR, "sample_book_page.png"))


def make_header_line() -> None:
    """A single text line, like a license plate or document header -> --psm 7."""
    w, h = 700, 220
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(FONT_BOLD, 64)
    text = "DECODE-LABS-4X7"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([(30, 30), (w - 30, h - 30)], outline=(20, 20, 20), width=4)
    draw.text(((w - tw) / 2, (h - th) / 2 - 10), text, font=font, fill=(10, 10, 10))

    final = _add_noise_and_shadow(img, skew_deg=-1.6)
    final.convert("RGB").save(os.path.join(OUT_DIR, "sample_header_line.png"))


def main() -> None:
    np.random.seed(42)
    make_invoice()
    make_book_page()
    make_header_line()
    print("Generated OCR sample images in:", OUT_DIR)
    for f in ["sample_invoice.png", "sample_book_page.png", "sample_header_line.png"]:
        print("  -", f)


if __name__ == "__main__":
    main()
