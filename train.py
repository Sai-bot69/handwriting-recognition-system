"""
Trains the CNN on MNIST and saves the model + a training curve plot.

Usage:
    python src/train.py
"""

import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import load_and_preprocess_data, build_augmenter
from model import build_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def train(epochs=10, batch_size=128):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    datagen = build_augmenter()
    datagen.fit(x_train)

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
    )

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "mnist_cnn.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save training curve
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plot_path = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
    plt.savefig(plot_path)
    print(f"Accuracy curve saved to {plot_path}")

    return model, history


if __name__ == "__main__":
    train()
