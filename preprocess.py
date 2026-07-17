"""
Data loading and preprocessing for the handwriting recognition CNN.

Loads the MNIST digit dataset, normalizes pixel values, reshapes for
CNN input, one-hot encodes labels, and sets up data augmentation.
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_and_preprocess_data():
    """
    Loads MNIST, normalizes, reshapes, and one-hot encodes.

    Returns:
        (x_train, y_train), (x_test, y_test): preprocessed train/test splits
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add channel dimension: CNNs expect (batch, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels (digits 0-9)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)


def build_augmenter():
    """
    Builds a Keras ImageDataGenerator for light augmentation.

    Rotation/shift/zoom ranges are kept small since digits are
    sensitive to large distortions (e.g. a rotated 6 can look like a 9).
    """
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    return datagen


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print(f"Train shape: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test shape:  {x_test.shape}, labels: {y_test.shape}")
