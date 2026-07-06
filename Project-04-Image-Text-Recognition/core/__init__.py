"""
Core package for Project 4: Building the Machine's Optic Nerve.

Modules:
    preprocessing     -> Step 1-4 Logic Skeleton (grayscale, blur, deskew, threshold)
    confidence_gate   -> The 80% Confidence Filter (softmax gate)
    ocr_engine        -> Path 1: OCR (pytesseract)
    object_detector   -> Path 2: Object Detection (cv2.dnn + MobileNet-SSD)
"""
