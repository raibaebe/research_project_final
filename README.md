# Sign Language Gesture Recognition

CNN-based recognition of 24 static ASL hand gestures (A-I, K-Y) trained on Sign Language MNIST, with real-time webcam prediction using MediaPipe hand detection.

## Project Structure

```
sign_language_ml/
├── data/
│   ├── sign_mnist_train.csv
│   ├── sign_mnist_test.csv
│   └── collected/            # Webcam-collected training data
├── models/
│   ├── trained_model.h5
│   └── confusion_matrix.png
├── src/
│   ├── data_loader.py        # MNIST CSV + collected data loading
│   ├── preprocessing.py      # Normalization, reshaping, augmentation
│   ├── model.py              # CNN architecture
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Metrics and confusion matrix
│   ├── landmark_extractor.py # MediaPipe hand detection
│   ├── predict.py            # Webcam and image prediction
│   └── collect.py            # Webcam data collection tool
├── main.py                   # CLI entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download Sign Language MNIST from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and place the CSV files in `data/`.

## Usage

### Train

```bash
python main.py train
```

Options: `--epochs 30`, `--batch-size 64`, `--no-augmentation`, `--model-path models/trained_model.h5`

Automatically merges MNIST data with any webcam-collected data in `data/collected/`.

### Evaluate

```bash
python main.py evaluate
```

Prints accuracy, per-class precision/recall/F1 scores, and saves a confusion matrix to `models/confusion_matrix.png`.

### Predict (webcam)

```bash
python main.py predict
```

Options: `--camera 0`, `--smoothing 7`

Press **q** or **ESC** to quit.

### Predict (image)

```bash
python main.py predict-image path/to/image.png
```

### Collect webcam data

```bash
python main.py collect
```

Options: `--output-dir data/collected`, `--camera 0`, `--frames 5`

Show a hand gesture and press the corresponding letter key to capture frames. Press **q** or **ESC** to quit. Collected images are saved as 28x28 grayscale PNGs and merged with MNIST data during training.

## Model

```
Conv2D(32) -> BN -> MaxPool -> Dropout(0.25)
Conv2D(64) -> BN -> MaxPool -> Dropout(0.25)
Dense(256) -> BN -> Dropout(0.5) -> Dense(25, softmax)
```

Input: 28x28 grayscale. Output: 25 classes (A-I, K-Y).

## Dataset

- **Sign Language MNIST** — 27,455 train / 7,172 test images, 28x28 grayscale, 24 letter classes (J and Z excluded — they require motion)
- **Collected data** — optional webcam captures to improve real-world accuracy
