"""CNN model for Sign Language MNIST classification."""

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,
)
from src.data_loader import NUM_CLASSES, IMG_SIZE


def build_cnn():
    model = Sequential([
        Conv2D(32, 3, activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, 3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Conv2D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_trained_model(model, filepath="models/trained_model.h5"):
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_trained_model(filepath="models/trained_model.h5"):
    """Load model from disk."""
    model = load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model
