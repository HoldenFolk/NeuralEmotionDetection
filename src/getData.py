import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


def load_fer2013_data(fer2013_path):
    data = pd.read_csv(fer2013_path)

    # Extract pixels and convert to numpy array
    pixels = data["pixels"].apply(
        lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()]
    )
    pixels = np.array(pixels.tolist(), dtype=np.float32).reshape(-1, 48, 48, 1)

    # Normalize pixel values to [0, 1]
    pixels = pixels / 255.0

    # One-hot encode labels
    labels = to_categorical(data["emotion"])

    # Split data into training, validation, and test set
    train_pixels = pixels[data["Usage"] == "Training"]
    val_pixels = pixels[data["Usage"] == "PublicTest"]
    test_pixels = pixels[data["Usage"] == "PrivateTest"]

    train_labels = labels[data["Usage"] == "Training"]
    val_labels = labels[data["Usage"] == "PublicTest"]
    test_labels = labels[data["Usage"] == "PrivateTest"]

    return train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels


if __name__ == "__main__":
    # Example usage
    fer2013_path = "assets/fer2013.csv"
    x_train, y_train, x_val, y_val, x_test, y_test = load_fer2013_data(fer2013_path)

    print(f"X Training Set: \n{x_train}\n")
    print(f"Y Training Set: \n{y_train}")
