"""Load Sign Language MNIST dataset and collected webcam data."""

import os
import numpy as np
import pandas as pd

# Label 0-24 -> letter A-Y (skipping J at index 9)
LABEL_TO_LETTER = {i: chr(ord('A') + i + (1 if i >= 9 else 0)) for i in range(25)}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}
NUM_CLASSES = 25
IMG_SIZE = 28


def load_csv(filepath):
    """Load a Sign Language MNIST CSV. Returns (images, labels)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Not found: {filepath}\n"
            f"Download from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist"
        )
    df = pd.read_csv(filepath)
    labels = df.iloc[:, 0].values.astype(np.int32)
    images = df.iloc[:, 1:].values.astype(np.float32).reshape(-1, IMG_SIZE, IMG_SIZE)
    return images, labels


def load_dataset(data_dir="data"):
    """Load train and test splits. Returns dict with images and labels."""
    print("Loading training data...")
    train_images, train_labels = load_csv(os.path.join(data_dir, "sign_mnist_train.csv"))
    print(f"  {len(train_images)} training samples")

    print("Loading test data...")
    test_images, test_labels = load_csv(os.path.join(data_dir, "sign_mnist_test.csv"))
    print(f"  {len(test_images)} test samples")

    return {
        "train_images": train_images, "train_labels": train_labels,
        "test_images": test_images, "test_labels": test_labels,
    }


def load_collected_data(collected_dir="data/collected"):
    """Load webcam-collected 28x28 grayscale PNGs from <dir>/<LETTER>/ folders."""
    import cv2

    if not os.path.isdir(collected_dir):
        return None

    images, labels = [], []
    for letter, label_idx in LETTER_TO_LABEL.items():
        class_dir = os.path.join(collected_dir, letter)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = cv2.imread(os.path.join(class_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            images.append(img.astype(np.float32))
            labels.append(label_idx)

    if not images:
        return None

    print(f"  Loaded {len(images)} collected webcam images")
    return np.array(images), np.array(labels, dtype=np.int32)


def label_to_letter(label):
    """Convert numeric label (0-24) to letter."""
    return LABEL_TO_LETTER.get(label, "?")
