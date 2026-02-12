import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

ROI_SIZE = 128


def extract_roi(image: np.ndarray, bbox: dict, save_path: str) -> dict:
    """
    Extract Region of Interest from detected palm area.

    Args:
        image: BGR image (OpenCV format)
        bbox: dict with x, y, w, h
        save_path: Path to save ROI image

    Returns:
        dict with 'roi_image' (numpy array), 'roi_image_path', and 'success'
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    # Crop palm region
    palm_crop = image[y : y + h, x : x + w]

    if palm_crop.size == 0:
        logger.error("Empty ROI crop")
        return {"success": False, "error": "Empty ROI region"}

    # Calculate center crop for ROI (inner palm region)
    crop_h, crop_w = palm_crop.shape[:2]
    cx, cy = crop_w // 2, crop_h // 2
    half = min(crop_w, crop_h) // 3

    roi = palm_crop[
        max(0, cy - half) : cy + half,
        max(0, cx - half) : cx + half,
    ]

    # Resize to standard size
    roi_resized = cv2.resize(roi, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)

    # Convert to grayscale for consistency
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi_gray)

    # Normalize
    roi_normalized = cv2.normalize(roi_enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # Save grayscale ROI
    cv2.imwrite(save_path, roi_normalized)
    logger.info(f"ROI extracted and saved: {ROI_SIZE}x{ROI_SIZE}")

    # Return 3-channel version for CNN input
    roi_3ch = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2BGR)

    return {
        "success": True,
        "roi_image": roi_3ch,
        "roi_image_path": save_path,
    }
