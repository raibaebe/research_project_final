"""Hand landmark extraction using MediaPipe Tasks API."""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
from src.data_loader import IMG_SIZE

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "hand_landmarker.task")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Pairs of landmark indices to draw as connections
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17),
]


def _ensure_model():
    """Download hand landmarker model if needed."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"Downloading hand landmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Done.")
    return _MODEL_PATH


class HandLandmarkExtractor:
    """Detects hands and extracts landmarks + bounding boxes from frames."""

    def __init__(self, max_num_hands=1, min_detection_confidence=0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model()),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def extract_landmarks(self, frame_rgb):
        """Detect hands in RGB frame. Returns list of {landmarks, bbox} or None."""
        result = self.landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        )
        if not result.hand_landmarks:
            return None

        h, w = frame_rgb.shape[:2]
        hands = []
        for hand_lms in result.hand_landmarks:
            coords = np.array([(lm.x, lm.y, lm.z) for lm in hand_lms], dtype=np.float32)
            xs, ys = coords[:, 0] * w, coords[:, 1] * h
            margin = 20
            bbox = (
                max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
                min(w, int(xs.max()) + margin), min(h, int(ys.max()) + margin),
            )
            hands.append({"landmarks": coords, "bbox": bbox})
        return hands

    def crop_hand_region(self, frame, bbox):
        """Crop and resize hand region to 28x28 grayscale."""
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    def draw_landmarks(self, frame, landmarks):
        """Draw hand skeleton on BGR frame."""
        h, w = frame.shape[:2]
        for i, j in _HAND_CONNECTIONS:
            pt1 = (int(landmarks[i, 0] * w), int(landmarks[i, 1] * h))
            pt2 = (int(landmarks[j, 0] * w), int(landmarks[j, 1] * h))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
        for lm in landmarks:
            cv2.circle(frame, (int(lm[0] * w), int(lm[1] * h)), 3, (0, 255, 0), -1)

    def close(self):
        self.landmarker.close()
