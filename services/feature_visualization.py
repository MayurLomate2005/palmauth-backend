import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def visualize_features(roi_gray_path: str, save_path: str) -> dict:
    """
    Apply edge detection and Gabor filters to visualize palm line features.

    Args:
        roi_gray_path: Path to the grayscale ROI image
        save_path: Path to save feature visualization

    Returns:
        dict with 'feature_image_path' and 'success'
    """
    roi = cv2.imread(roi_gray_path, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        return {"success": False, "error": "Could not read ROI image"}

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (5, 5), 1.0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Gabor filter bank for line detection at multiple orientations
    gabor_sum = np.zeros_like(roi, dtype=np.float64)
    for theta in np.arange(0, np.pi, np.pi / 8):
        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=3.0,
            theta=theta,
            lambd=8.0,
            gamma=0.5,
            psi=0,
        )
        filtered = cv2.filter2D(roi, cv2.CV_64F, kernel)
        gabor_sum += np.abs(filtered)

    # Normalize Gabor response
    gabor_norm = cv2.normalize(gabor_sum, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Combine edges and Gabor features
    combined = cv2.addWeighted(edges, 0.5, gabor_norm, 0.5, 0)

    # Create colored visualization
    feature_vis = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    feature_vis[:, :, 0] = 0  # B
    feature_vis[:, :, 1] = combined  # G - palm lines in green
    feature_vis[:, :, 2] = edges  # R - edges in red

    cv2.imwrite(save_path, feature_vis)
    logger.info("Feature visualization saved")

    return {
        "success": True,
        "feature_image_path": save_path,
    }
