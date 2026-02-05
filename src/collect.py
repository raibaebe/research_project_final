"""Webcam data collection for Sign Language MNIST."""

import os
import time

import cv2

from src.data_loader import LETTER_TO_LABEL
from src.landmark_extractor import HandLandmarkExtractor


def run_collection(output_dir="data/collected", camera_index=0,
                    frames_per_capture=5):
    """Run webcam data collection. Press letter key to capture, 'q'/ESC to quit."""
    extractor = HandLandmarkExtractor()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    valid_keys = {ord(l.lower()): l for l in LETTER_TO_LABEL}

    counts = {}
    for letter in LETTER_TO_LABEL:
        d = os.path.join(output_dir, letter)
        counts[letter] = len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.isdir(d) else 0

    total_saved = sum(counts.values())
    current_label = ""
    last_capture_time = 0

    print(f"Data collection: {total_saved} existing images in {output_dir}")
    print(f"Press letter key to capture ({frames_per_capture} frames each), 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands
            hands = extractor.extract_landmarks(frame_rgb)
            hand_crop = None

            if hands:
                hand = hands[0]
                hand_crop = extractor.crop_hand_region(frame, hand["bbox"])
                extractor.draw_landmarks(frame, hand["landmarks"])
                x1, y1, x2, y2 = hand["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw UI
            h, w = frame.shape[:2]

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            status = "Hand detected" if hands else "No hand detected"
            status_color = (0, 255, 0) if hands else (0, 0, 255)
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.putText(frame, f"Total collected: {total_saved}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if current_label:
                cv2.putText(frame, f"Last: {current_label} ({counts.get(current_label, 0)} imgs)",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.putText(frame, "Press letter key to capture | q to quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Show 28x28 preview enlarged in the corner
            if hand_crop is not None:
                preview = cv2.resize(hand_crop, (100, 100), interpolation=cv2.INTER_NEAREST)
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                frame[5:105, w - 105:w - 5] = preview_bgr

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            if key in valid_keys and hand_crop is not None:
                now = time.time()
                if now - last_capture_time < 0.1:
                    continue
                last_capture_time = now

                letter = valid_keys[key]
                current_label = letter
                class_dir = os.path.join(output_dir, letter)
                os.makedirs(class_dir, exist_ok=True)

                saved = 0
                for _ in range(frames_per_capture):
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    frame2 = cv2.flip(frame2, 1)
                    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    hands2 = extractor.extract_landmarks(frame2_rgb)
                    img = extractor.crop_hand_region(frame2, hands2[0]["bbox"]) if hands2 else hand_crop

                    idx = counts.get(letter, 0)
                    cv2.imwrite(os.path.join(class_dir, f"collected_{idx:04d}.png"), img)
                    counts[letter] = idx + 1
                    total_saved += 1
                    saved += 1

                print(f"  Saved {saved} frames for '{letter}' "
                      f"(total for class: {counts[letter]})")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print(f"\nCollection complete. Total images saved: {total_saved}")
        print(f"Saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    run_collection()
