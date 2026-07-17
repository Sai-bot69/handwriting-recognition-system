# Handwriting Recognition System (CNN + OCR)

A handwritten digit classifier built with a Convolutional Neural Network
(trained on MNIST), paired with a Tesseract OCR pipeline for extracting
full handwritten sentences from scanned images.

> Originally developed as a final-year B.E. project at ATME College of
> Engineering by a 4-member team (Manjula P, Mohammed Akif, Spoorthi V M,
> Sai Ram). This repository contains a cleaned-up, restructured version of
> the digit-recognition and OCR components I worked on.

## What this does

- **Digit classification (CNN):** trained on MNIST, reaches ~99% test
  accuracy on isolated handwritten digits (0–9).
- **Free-text OCR (Tesseract):** extracts text from images of full
  handwritten sentences. This is a separate, non-neural pipeline — it is
  noticeably less accurate than the digit classifier on messy or cursive
  handwriting (see Limitations below).

## Project structure

```
handwriting-recognition-system/
├── src/
│   ├── preprocess.py   # MNIST loading, normalization, augmentation
│   ├── model.py         # CNN architecture
│   ├── train.py         # Training loop, saves model + accuracy plot
│   ├── predict.py       # Predict a single digit from an image
│   └── ocr_utils.py     # Tesseract-based sentence/word OCR
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd handwriting-recognition-system
pip install -r requirements.txt
```

OCR also requires the Tesseract binary itself (not just the Python
wrapper):

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

**Train the digit classifier:**
```bash
python src/train.py
```
Saves the trained model to `models/mnist_cnn.keras` and a training-curve
plot to `outputs/accuracy_curve.png`.

**Predict a single digit:**
```bash
python src/predict.py --image path/to/digit.png
```

**Run OCR on a handwritten note/sentence:**
```bash
python src/ocr_utils.py --image path/to/note.jpg
```

## Results

- CNN test accuracy on MNIST: **~99.2%**
- OCR performance on free-form cursive handwriting: noticeably lower —
  Tesseract struggles with connected/cursive letterforms and works best
  on clear block print.

## Limitations & honest notes

- The CNN only classifies **isolated single digits** (MNIST). It does not
  recognize letters, words, or full sentences — that part is handled
  separately by Tesseract OCR, which uses no learned model from this
  project and is meaningfully weaker on handwriting than on printed text.
- No custom/EMNIST dataset is used in this implementation, despite being
  mentioned in earlier drafts of the project report — the training
  pipeline here is MNIST-only.
- Future work: train directly on EMNIST (letters + digits) or a
  handwriting-specific dataset (e.g. IAM), and explore CRNN + CTC loss
  for genuine end-to-end sentence recognition instead of relying on
  Tesseract.

## References

- LeCun et al., *Gradient-Based Learning Applied to Document
  Recognition*, 1998
- Smith, *An Overview of the Tesseract OCR Engine*, 2007
- Graves et al., *Connectionist Temporal Classification*, 2006
