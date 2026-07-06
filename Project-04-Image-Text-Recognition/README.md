# Project 4: Building the Machine's Optic Nerve
### The DecodeLabs Architect's Playbook for Image & Text Recognition

> *"Mastering Project 4 means building the bridge between the physical world
> and computational logic. You are graduating to machine perception."*

A complete, production-style Python recognition pipeline that implements
**both** execution paths described in the Project 4 briefing:

| | Path 1 — OCR | Path 2 — Object Detection |
|---|---|---|
| **Core Objective** | Extracting machine-readable strings | Identifying & locating physical entities |
| **Primary Library** | `pytesseract` | `cv2.dnn` + MobileNet-SSD |
| **Pre-Processing** | Grayscale, blur, deskew, adaptive thresholding | 4D Blob construction (`blobFromImage`) |
| **Output** | Formatted text strings + confidence | `(X, Y, W, H)` bounding boxes + labels |

---

## Table of Contents
1. [The Paradigm Shift](#the-paradigm-shift)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Path 1: OCR — Deep Dive](#path-1-ocr--deep-dive)
6. [Path 2: Object Detection — Deep Dive](#path-2-object-detection--deep-dive)
7. [The Logic Skeleton (Pre-Processing)](#the-logic-skeleton-pre-processing)
8. [The 80% Confidence Gate](#the-80-confidence-gate)
9. [The Gatekeeper Rule (Milestone Validation)](#the-gatekeeper-rule-milestone-validation)
10. [Sample Images](#sample-images)
11. [CLI Reference](#cli-reference)
12. [Troubleshooting](#troubleshooting)
13. [Credits & Model Provenance](#credits--model-provenance)

---

## The Paradigm Shift

For decades, computing meant **structured data** — spreadsheets, databases,
clean CSVs. That world accounts for less than 20% of global enterprise
data. The frontier is **unstructured data**: scanned documents, video
feeds, and raw images — over 80% of everything an enterprise actually
holds.

To a machine, an image is never "a picture." It is a massive
three-dimensional array:

```
Height (H) × Width (W)  ->  spatial pixel resolution
Depth (C)                ->  3 color channels (Red, Green, Blue)
Intensity                ->  every pixel channel holds a value 0–255
```

A single 512×512 image generates **786,432 distinct data points**.
Altering a single coordinate directly alters the machine's reality. This
project builds the pipeline that reads that array and turns it into
trustworthy, machine-readable intelligence — the machine's **optic nerve**.

---

## Project Structure

```
project4_optic_nerve/
├── main.py                        # Single CLI entry point (interactive + flags)
├── check_environment.py           # Preflight diagnostic — run this FIRST
├── validate.py                    # Gatekeeper Rule — automated 4-milestone check
├── demo_preprocessing_stages.py   # Saves each pre-processing stage as its own image
├── config.py                      # Centralized constants (paths, thresholds)
├── requirements.txt
├── README.md                      # You are here
│
├── core/                          # The engine room
│   ├── __init__.py
│   ├── preprocessing.py           # Step 1-4: grayscale, blur, deskew, threshold
│   ├── confidence_gate.py         # The 80% Confidence Filter (softmax gate)
│   ├── ocr_engine.py              # Path 1: pytesseract pipeline
│   └── object_detector.py         # Path 2: cv2.dnn + MobileNet-SSD pipeline
│
├── models/
│   ├── download_models.py         # Fetches MobileNet-SSD weights (~23 MB)
│   ├── MobileNetSSD_deploy.prototxt     # Network architecture (bundled)
│   └── MobileNetSSD_deploy.caffemodel   # Pre-trained weights (bundled)
│
├── sample_images/
│   ├── generate_samples.py            # Regenerates the 3 synthetic OCR images
│   ├── sample_invoice.png             # Sparse/scattered layout -> --psm 11
│   ├── sample_book_page.png           # Uniform paragraph block -> --psm 6
│   ├── sample_header_line.png         # Single text line -> --psm 7
│   ├── sample_object_dog_bike_car.jpg # dog + bicycle + car (Path 2 demo)
│   ├── sample_object_person.jpg       # person + dog (Path 2 demo)
│   └── sample_object_horses.jpg       # horses (Path 2 demo)
│
├── output/                        # All generated results land here (git-ignored)
│   └── stages/                    # Intermediate pre-processing snapshots
│
└── tests/
    └── test_pipeline.py            # 14 sanity checks across both paths
```

---

## Installation

**Requirements:** Python 3.9+ and the Tesseract OCR *system binary*
(pytesseract is only a wrapper — it does not bundle the engine itself).

```bash
# 1. Clone / unzip the project, then enter it
cd project4_optic_nerve

# 2. Install the Tesseract OCR engine (OS-level, one time)
#    Ubuntu / Debian:
sudo apt-get update && sudo apt-get install -y tesseract-ocr
#    macOS (Homebrew):
brew install tesseract
#    Windows: install from https://github.com/UB-Mannheim/tesseract/wiki

# 3. Verify it's on PATH
tesseract --version

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. (Optional) The MobileNet-SSD model files are already bundled under
#    models/. If they are ever missing/corrupted, re-fetch them with:
python models/download_models.py
```

---

## Quick Start

**Step 0 — Run the preflight check first (especially on Windows):**

```bash
python check_environment.py
```

This single command verifies your Python version, all required
packages, the Tesseract OCR *engine binary* (the #1 source of setup
issues — see [Troubleshooting](#troubleshooting)), the bundled
MobileNet-SSD model files, and all sample images — each with a
pass/fail line and, on failure, the exact fix. Don't run `main.py`
until this shows `10/10 checks passed`.

**Interactive menu** (recommended for first-time use):

```bash
python main.py
```

This opens a guided menu — pick Path 1 (OCR), Path 2 (Object Detection),
or run both back-to-back on the bundled sample images.

**Direct CLI usage:**

```bash
# Path 1 — OCR on the sample invoice, using sparse-text mode
python main.py --path ocr --image sample_images/sample_invoice.png --psm 11

# Path 2 — Object Detection on the sample dog/bike/car photo
python main.py --path detect --image sample_images/sample_object_dog_bike_car.jpg

# Run BOTH paths on the same image in one command
python main.py --path both --image sample_images/sample_invoice.png
```

**Validate every Gatekeeper milestone end-to-end:**

```bash
python validate.py
```

**Run the automated test suite:**

```bash
python tests/test_pipeline.py
```

**Visualize every pre-processing stage as its own saved image:**

```bash
python demo_preprocessing_stages.py sample_images/sample_invoice.png
```

Every run writes its annotated, "visual confirmation" output into
`output/`, e.g. `output/sample_invoice_ocr_annotated.png` or
`output/sample_object_dog_bike_car_detect_annotated.png`.

---

## Path 1: OCR — Deep Dive

**The Engine:** `pytesseract` wraps Google's Tesseract engine, which
internally runs a convolutional + bi-directional LSTM pipeline to read
sequences of characters.

### Tuning the PSM (Page Segmentation Mode)

Layout configuration is critical for accuracy. This project exposes all
four modes called out in the briefing:

| Flag | Mode | Best for |
|---|---|---|
| `--psm 3` | Fully automatic (default) | Varied, unknown layouts |
| `--psm 6` | Single uniform block of text | Book pages, paragraphs |
| `--psm 7` | Single text line | Number plates, headers |
| `--psm 11` | Sparse, scattered text | Invoices, forms |

```bash
python main.py --path ocr --image sample_images/sample_book_page.png --psm 6
python main.py --path ocr --image sample_images/sample_header_line.png --psm 7
```

### What you get back

- Every recognized word, with its own confidence score (0–100%)
- The full reconstructed text string
- A `PreprocessingReport` proving grayscale + adaptive thresholding ran
- An annotated PNG: **green boxes** = kept (≥80% confidence), **red
  boxes** = dropped (below the gate)

---

## Path 2: Object Detection — Deep Dive

**The Backbone:** MobileNet v3, using depthwise separable convolutions
to filter input channels separately — optimized for high-speed,
real-time inference on edge devices with minimal compute.

**Transfer Learning:** rather than training from scratch, this project
downloads a MobileNet-SSD model pre-trained on the Pascal VOC dataset
(millions of ImageNet-derived visual concepts already baked in), then
plugs a ready-made detection head onto it — "downloading a degree"
instead of re-teaching the machine what an edge or a gradient is.

### Step 1: Blob Construction

```python
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0 / 127.5,
    size=(300, 300),          # required network input dimensions
    mean=(127.5, 127.5, 127.5),  # mean subtraction
    swapRB=False,
    crop=False,
)
```

### Decoding the Matrix: Anatomy of a Bounding Box

The network does **not** output an image — it outputs normalized `(X, Y,
W, H)` coordinates in the `[0, 1]` range. This project multiplies those
normalized values by the original image's actual pixel width/height to
physically place the bounding box overlay ("Coordinate Scaling").

### Supported classes (Pascal VOC, 20 classes)

`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor`

```bash
python main.py --path detect --image sample_images/sample_object_person.jpg
```

---

## The Logic Skeleton (Pre-Processing)

Both paths share the exact same pre-processing module
(`core/preprocessing.py`) so "Pre-Processing Integrity" is guaranteed
identically everywhere:

1. **Grayscale Conversion** — collapses the 3D RGB matrix into a 1D
   intensity matrix, removing distracting color data.
2. **Gaussian Blur** — smooths micro-imperfections and sensor noise.
3. **Deskewing** — calculates the rotation angle of the dominant
   content block via `cv2.minAreaRect` and rotates the image back to a
   perfect horizontal baseline.
4. **Adaptive Thresholding (Otsu's Method)** — forces every pixel to
   commit to black or white:

   ```
   IF pixel_intensity >= cutoff: pixel = 255 (white)
   IF pixel_intensity <  cutoff: pixel = 0   (black)
   ```

   The cutoff is calculated automatically per image (Otsu), not
   hard-coded, so it adapts to each image's own lighting conditions.

Run `python demo_preprocessing_stages.py <image>` to save all five
snapshots (original → grayscale → blurred → deskewed → binary) as
individual PNGs under `output/stages/`.

---

## The 80% Confidence Gate

AI does not *know* what a character or object is — it calculates a
statistical probability (softmax output) of what it might be. Without a
filter, the model treats every guess with equal certainty, leading to
confident hallucinations and false positives.

`core/confidence_gate.py` implements the exact rule from the briefing:

```python
if confidence >= 0.80:
    draw_box_and_label()
else:
    drop_detection()
```

High thresholds minimize false positives but increase the risk of false
negatives — **80% is the absolute minimum standard for Project 4**, and
every pipeline run reports both what was kept *and* what was dropped so
you can see the trade-off happening in real time.

---

## The Gatekeeper Rule (Milestone Validation)

`validate.py` runs both paths end-to-end against the bundled sample
images and checks all four required milestones automatically:

| # | Milestone | What is checked |
|---|---|---|
| 1 | **Library Integration** | `pytesseract` / `cv2.dnn` load and run without error |
| 2 | **Pre-Processing Integrity** | Grayscale + Adaptive Thresholding demonstrably executed |
| 3 | **Accuracy Benchmarking** | Best confidence score on the final output ≥ 80% |
| 4 | **Visual Confirmation** | A legible OCR string / accurately labeled bounding-box image is generated |

```bash
$ python validate.py
...
========================================================================
 FINAL RESULT
========================================================================
  Path 1 (OCR)              : ✅ PASS
  Path 2 (Object Detection) : ✅ PASS

All Gatekeeper milestones passed. Project 4 requirements met.
```

---

## Sample Images

No external downloads are required to try the project — everything
ships in `sample_images/`:

**OCR path** (synthetically generated with realistic noise, shadows,
and a slight camera-skew, via `sample_images/generate_samples.py`):
- `sample_invoice.png` — scattered invoice layout → `--psm 11`
- `sample_book_page.png` — uniform paragraph → `--psm 6`
- `sample_header_line.png` — single bold line → `--psm 7`

**Object Detection path** (classic, freely-distributable computer-vision
benchmark photographs):
- `sample_object_dog_bike_car.jpg` — dog, bicycle, and car
- `sample_object_person.jpg` — person and dog
- `sample_object_horses.jpg` — multiple horses

Want fresh synthetic OCR samples? Regenerate them any time:

```bash
python sample_images/generate_samples.py
```

Want to test on your own image? Just point `--image` at any file path —
both paths accept standard JPG/PNG input.

---

## CLI Reference

```
python main.py [--path {ocr,detect,both}] [--image IMAGE]
               [--psm {3,6,7,11}] [--lang LANG] [--no-deskew]

  --path        Which recognition path to run.
                 Omit entirely to launch the interactive menu.
  --image       Path to the input image (required if --path is set).
  --psm         Tesseract Page Segmentation Mode (OCR path only).
                 3 = auto, 6 = block, 7 = single line, 11 = sparse.
                 Default: 3.
  --lang        Tesseract language pack, e.g. 'eng', 'fra'. Default: eng.
  --no-deskew   Disable the deskew pre-processing step (OCR path only).
```

---

## Troubleshooting

**`TesseractNotConfiguredError` / `TesseractNotFoundError` / "the system cannot find the file specified" (Windows)**

This is the single most common setup issue and is **not a bug in the
code** — it means the Tesseract OCR *engine* (a separate program) isn't
installed, or is installed but not visible on your PATH. `pytesseract`
is only a thin Python wrapper around that program; pip installing
`pytesseract` never installs the engine itself.

This project auto-detects the engine in the most common install
locations, and if it truly can't be found, it now fails with a clear,
actionable message (instead of a raw Python traceback) telling you
exactly what to do. To fix it on Windows:

1. Download and run the installer:
   https://github.com/UB-Mannheim/tesseract/wiki
2. Note the install folder shown by the installer
   (default: `C:\Program Files\Tesseract-OCR`).
3. Either:
   - **Add it to PATH** — Windows Settings → "Edit the system
     environment variables" → Environment Variables → select `Path`
     → New → paste the install folder → OK everywhere → **open a
     brand-new terminal** (PATH changes never apply to already-open
     terminals), or
   - **Point to it directly in code** — add this near the top of
     `main.py` (or anywhere before the first OCR call):
     ```python
     import pytesseract
     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```
4. Verify the engine is visible with:
   ```
   tesseract --version
   ```
5. Then re-run the preflight check to confirm:
   ```
   python check_environment.py
   ```

macOS: `brew install tesseract`. Linux: `sudo apt-get install -y tesseract-ocr`.
Path 2 (Object Detection) does not depend on Tesseract at all and will
work regardless.

**`FileNotFoundError: MobileNet-SSD model files not found`**
Run `python models/download_models.py` to fetch
`MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel`
into the `models/` folder. (Both are already bundled in this delivery,
so this should only happen if they were deleted.)

**OCR confidence looks low on my own image**
Try a different `--psm` mode that matches your layout (see the table
above), and make sure the source image has decent resolution/lighting —
the pre-processing pipeline helps, but it can't recover text that's too
blurry or too small to begin with.

**Object detector finds nothing / low confidence**
MobileNet-SSD (Pascal VOC) only recognizes the 20 classes listed above.
If your image contains other kinds of objects, the model has no label
for them — this is expected model behavior, not a bug.

---

## Credits & Model Provenance

- **Tesseract OCR** — Google / Tesseract OCR project.
- **MobileNet-SSD (Caffe, Pascal VOC-trained)** — architecture and
  pre-trained weights from the widely-used
  [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)
  community release.
- **Object-detection sample photographs** (`dog.jpg`, `person.jpg`,
  `horses.jpg`) — the classic, freely redistributed computer-vision
  benchmark images originally bundled with the Darknet/YOLO project.
- **OCR sample images** — synthetically generated for this project via
  `sample_images/generate_samples.py` (Pillow), no external content used.

---

*Built for DecodeLabs — Project 4: Image or Text Recognition (Basic).
Keep innovating, keep learning, keep building the future.*
