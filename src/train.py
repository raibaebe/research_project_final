"""Training pipeline for the Sign Language CNN."""

import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.data_loader import load_dataset, load_collected_data
from src.preprocessing import preprocess_pipeline, create_augmenter
from src.model import build_cnn, save_trained_model


def train(
    data_dir="data",
    model_path="models/trained_model.h5",
    epochs=30,
    batch_size=64,
    use_augmentation=True,
    validation_split=0.1,
):
    dataset = load_dataset(data_dir)
    train_images = dataset["train_images"]
    train_labels = dataset["train_labels"]

    # Merge
    collected = load_collected_data(os.path.join(data_dir, "collected"))
    if collected is not None:
        train_images = np.concatenate([train_images, collected[0]])
        train_labels = np.concatenate([train_labels, collected[1]])
        print(f"  Merged total: {len(train_images)} training images")

    X_train, y_train = preprocess_pipeline(train_images, train_labels)
    X_test, y_test = preprocess_pipeline(dataset["test_images"], dataset["test_labels"])

    # Split validation set
    num_val = int(len(X_train) * validation_split)
    idx = np.random.permutation(len(X_train))
    X_val, y_val = X_train[idx[:num_val]], y_train[idx[:num_val]]
    X_train, y_train = X_train[idx[num_val:]], y_train[idx[num_val:]]

    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = build_cnn()
    model.summary()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    if use_augmentation:
        aug = create_augmenter()
        aug.fit(X_train)
        history = model.fit(
            aug.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs, validation_data=(X_val, y_val),
            callbacks=callbacks, verbose=1,
        )
    else:
        history = model.fit(
            X_train, y_train, batch_size=batch_size,
            epochs=epochs, validation_data=(X_val, y_val),
            callbacks=callbacks, verbose=1,
        )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

    save_trained_model(model, model_path)
    return history.history
