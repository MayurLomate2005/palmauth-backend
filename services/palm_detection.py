import cv2
import numpy as np
import os
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)


# Initialize MediaPipe HandLandmarker once
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"  # Make sure this file exists
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)


def detect_palm(image: np.ndarray, save_path: str) -> dict:
    """
    Detect palm region using MediaPipe HandLandmarker (Tasks API).
    """

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    detection_result = hand_landmarker.detect(mp_image)

    if not detection_result.hand_landmarks:
        logger.warning("No hand detected in image")
        return {"success": False, "error": "No hand detected"}

    hand_landmarks = detection_result.hand_landmarks[0]

    # Convert normalized landmarks to pixel coordinates
    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]

    x_min = max(0, int(min(x_coords) * w) - 20)
    y_min = max(0, int(min(y_coords) * h) - 20)
    x_max = min(w, int(max(x_coords) * w) + 20)
    y_max = min(h, int(max(y_coords) * h) + 20)

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # Draw bounding box
    annotated = image.copy()
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

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
