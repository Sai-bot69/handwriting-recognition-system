"""
Loads a trained model and predicts a digit from a 28x28 grayscale image.

Usage:
    python src/predict.py --image path/to/digit.png
"""

import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_cnn.keras")


def preprocess_image(image_path):
    """Loads an arbitrary image, converts to 28x28 grayscale, normalizes."""
    img = Image.open(image_path).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def predict_digit(image_path, model_path=MODEL_PATH):
    model = load_model(model_path)
    sample = preprocess_image(image_path)
    prediction = model.predict(sample)
    return int(prediction.argmax()), float(prediction.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a handwritten digit.")
    parser.add_argument("--image", required=True, help="Path to a digit image")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to a trained .keras model")
    args = parser.parse_args()

    label, confidence = predict_digit(args.image, args.model)
    print(f"Predicted digit: {label} (confidence: {confidence:.2%})")
