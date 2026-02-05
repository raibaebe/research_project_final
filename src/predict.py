"""Real-time sign language prediction from webcam."""

import collections
import time

import cv2
import numpy as np

from src.data_loader import label_to_letter, IMG_SIZE
from src.preprocessing import preprocess_single_frame
from src.model import load_trained_model
from src.landmark_extractor import HandLandmarkExtractor


class PredictionSmoother:
    def __init__(self, window_size=7):
        self.history = collections.deque(maxlen=window_size)

    def update(self, label):
        self.history.append(label)
        return collections.Counter(self.history).most_common(1)[0][0]

    def reset(self):
        self.history.clear()


def run_webcam_prediction(model_path="models/trained_model.h5",
                          camera_index=0, smoothing_window=7):
    print("Loading model...")
    model = load_trained_model(model_path)
    extractor = HandLandmarkExtractor()
    smoother = PredictionSmoother(smoothing_window)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened. Press 'q' to quit, 'r' to reset.")

    letter, confidence, prev_time = "", 0.0, time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = extractor.extract_landmarks(frame_rgb)

            if hands:
                hand = hands[0]
                # Crop before drawing overlays
                hand_img = extractor.crop_hand_region(frame, hand["bbox"])
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                hand_img = clahe.apply(hand_img)
                input_tensor = preprocess_single_frame(hand_img)

                # Draw on display frame
                extractor.draw_landmarks(frame, hand["landmarks"])
                x1, y1, x2, y2 = hand["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                probs = model.predict(input_tensor, verbose=0)[0]
                pred_label = int(np.argmax(probs))
                confidence = float(probs[pred_label])
                letter = label_to_letter(smoother.update(pred_label))
            else:
                letter, confidence = "", 0.0

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Draw overlay
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            if letter:
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(frame, f"Prediction: {letter}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "No hand detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(frame, "q: quit | r: reset", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                smoother.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


def predict_single_image(model_path, image_path):
    """Predict letter from a single image file. Returns (letter, confidence)."""
    model = load_trained_model(model_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    probs = model.predict(preprocess_single_frame(img), verbose=0)[0]
    pred_label = int(np.argmax(probs))
    return label_to_letter(pred_label), float(probs[pred_label])
