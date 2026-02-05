"""Preprocessing: normalization, reshaping, augmentation."""

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data_loader import NUM_CLASSES, IMG_SIZE


def preprocess_pipeline(images, labels):
    """Normalize images to [0,1], reshape to (N,28,28,1), one-hot encode labels."""
    images = images.astype(np.float32) / 255.0
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return images, labels


def preprocess_single_frame(frame_gray):
    """Preprocess a single 28x28 grayscale frame for prediction."""
    img = frame_gray.astype(np.float32) / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def create_augmenter():
    """Create augmenter with mild geometric transforms."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
