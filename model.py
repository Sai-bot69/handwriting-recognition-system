"""
CNN architecture for handwritten digit recognition (MNIST, 28x28 grayscale).
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds and compiles the CNN.

    Architecture: two conv blocks (Conv2D -> BatchNorm -> MaxPool),
    followed by a dense classification head with dropout for
    regularization.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(0.001), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
