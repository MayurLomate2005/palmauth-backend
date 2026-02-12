import mediapipe as mp
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_palm(image: np.ndarray, save_path: str) -> dict:
    """
    Detect palm region using MediaPipe Hands.

    Args:
        image: BGR image (OpenCV format)
        save_path: Path to save the annotated image

    Returns:
        dict with 'bbox' (x, y, w, h), 'detected_image_path', and 'success' flag
    """
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        results = hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            logger.warning("No hand detected in image")
            return {"success": False, "error": "No hand detected"}

        hand_landmarks = results.multi_hand_landmarks[0]

        # Calculate bounding box from landmarks
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        x_min = max(0, int(min(x_coords) * w) - 20)
        y_min = max(0, int(min(y_coords) * h) - 20)
        x_max = min(w, int(max(x_coords) * w) + 20)
        y_max = min(h, int(max(y_coords) * h) + 20)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Draw bounding box and landmarks on image
        annotated = image.copy()
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        mp_drawing.draw_landmarks(
            annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Add label
        cv2.putText(
            annotated,
            "Palm Detected",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imwrite(save_path, annotated)
        logger.info(f"Palm detected, bbox: ({x_min},{y_min},{bbox_w},{bbox_h})")

        return {
            "success": True,
            "bbox": {"x": x_min, "y": y_min, "w": bbox_w, "h": bbox_h},
            "detected_image_path": save_path,
        }
