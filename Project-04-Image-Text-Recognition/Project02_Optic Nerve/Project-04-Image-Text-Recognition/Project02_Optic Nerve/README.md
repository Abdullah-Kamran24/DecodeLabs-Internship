# DecodeLabs Internship

A collection of projects completed as part of the DecodeLabs internship
program, each in its own folder with a self-contained README, setup
instructions, and sample data.

## Projects

| # | Project | Description | Tech Stack |
|---|---|---|---|
| 4 | [Image or Text Recognition (Basic)](./Project-04-Image-Text-Recognition) | A dual-path recognition pipeline: OCR (pytesseract) and Object Detection (OpenCV DNN + MobileNet-SSD), with a shared pre-processing pipeline and an 80% confidence gate. | Python, OpenCV, Tesseract OCR, MobileNet-SSD |

*(More projects will be added here as the internship progresses.)*

## Structure

Each project folder is fully self-contained:

```
DecodeLabs-Internship/
├── README.md                              <- you are here
└── Project-04-Image-Text-Recognition/
    ├── README.md                          <- detailed docs for this project
    ├── main.py
    ├── check_environment.py
    ├── validate.py
    ├── core/
    ├── models/
    ├── sample_images/
    └── tests/
```

To run any project, `cd` into its folder and follow that project's own
README.
