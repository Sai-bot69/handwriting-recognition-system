"""
OCR text extraction using Tesseract, for full sentences/words rather
than single digits (the CNN above only classifies isolated digits).

Requires the Tesseract binary to be installed separately:
    Ubuntu/Debian: sudo apt install tesseract-ocr
    macOS:         brew install tesseract
    Windows:       https://github.com/UB-Mannheim/tesseract/wiki

Usage:
    python src/ocr_utils.py --image path/to/handwritten_note.jpg
"""

import argparse
import cv2
import pytesseract
from PIL import Image

# NOTE: only set this if pytesseract can't find Tesseract automatically.
# Example (Windows): pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path):
    """Grayscale + upscale + Otsu threshold, to improve OCR legibility."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image


def extract_text_from_image(image_path, lang="eng"):
    processed_image = preprocess_image(image_path)
    pil_image = Image.fromarray(processed_image)
    extracted_text = pytesseract.image_to_string(pil_image, lang=lang)
    return extracted_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a handwritten image via OCR.")
    parser.add_argument("--image", required=True, help="Path to an image of handwritten text")
    args = parser.parse_args()

    text = extract_text_from_image(args.image)
    print("Extracted Text:")
    print(text)
