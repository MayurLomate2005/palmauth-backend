from flask import Blueprint, request, jsonify, current_app
import cv2
import numpy as np
import os
import uuid
import base64
import logging
from models import db, User
from services.palm_detection import detect_palm
from services.roi_extraction import extract_roi
from services.feature_visualization import visualize_features
from services.feature_extraction import extract_embedding

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


def decode_image(data: str) -> np.ndarray:
    """Decode base64 image data to OpenCV format."""
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def image_to_base64(path: str) -> str:
    """Convert image file to base64 data URL."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def process_pipeline(image: np.ndarray, uid: str, config) -> dict:
    """Run the full processing pipeline on an image."""
    # Step 1: Save original
    orig_path = os.path.join(config.UPLOAD_ORIGINAL, f"{uid}.jpg")
    cv2.imwrite(orig_path, image)

    # Step 2: Palm detection
    det_path = os.path.join(config.UPLOAD_DETECTED, f"{uid}.jpg")
    detection = detect_palm(image, det_path)
    if not detection["success"]:
        return {"success": False, "error": detection["error"]}

    # Step 3: ROI extraction
    roi_path = os.path.join(config.UPLOAD_ROI, f"{uid}.jpg")
    roi_result = extract_roi(image, detection["bbox"], roi_path)
    if not roi_result["success"]:
        return {"success": False, "error": roi_result["error"]}

    # Step 4: Feature visualization
    feat_path = os.path.join(config.UPLOAD_FEATURES, f"{uid}.jpg")
    feat_result = visualize_features(roi_path, feat_path)

    # Step 5: CNN embedding
    emb_result = extract_embedding(roi_result["roi_image"], config.EMBEDDING_DIM)
    if not emb_result["success"]:
        return {"success": False, "error": "Embedding extraction failed"}

    return {
        "success": True,
        "embedding": emb_result["embedding"],
        "images": {
            "original_image": image_to_base64(orig_path),
            "detected_image": image_to_base64(det_path),
            "roi_image": image_to_base64(roi_path),
            "feature_image": image_to_base64(feat_path) if feat_result["success"] else None,
        },
    }


@auth_bp.route("/api/register", methods=["POST"])
def register():
    """Register a new user with palm images."""
    try:
        data = request.get_json()

        name = data.get("name")
        roll_number = data.get("roll_number")
        images = data.get("images", [])  # List of base64 images

        if not name or not roll_number:
            return jsonify({"error": "Name and roll_number are required"}), 400

        if not images or len(images) == 0:
            return jsonify({"error": "At least one palm image is required"}), 400

        # Check duplicate
        existing = User.query.filter_by(roll_number=roll_number).first()
        if existing:
            return jsonify({"error": f"User with roll number {roll_number} already exists"}), 409

        config = current_app.config

        # Process each image and collect embeddings
        embeddings = []
        pipeline_result = None

        for i, img_data in enumerate(images):
            image = decode_image(img_data)
            uid = f"{roll_number}_{uuid.uuid4().hex[:8]}_{i}"

            result = process_pipeline(image, uid, current_app)
            if not result["success"]:
                logger.warning(f"Image {i} failed: {result.get('error')}")
                continue

            embeddings.append(result["embedding"])
            if pipeline_result is None:
                pipeline_result = result

        if len(embeddings) == 0:
            return jsonify({"error": "No valid palm images could be processed"}), 400

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = (np.array(avg_embedding) / norm).tolist()

        # Store user
        user = User(name=name, roll_number=roll_number, num_samples=len(embeddings))
        user.set_embedding(avg_embedding)
        db.session.add(user)
        db.session.commit()

        logger.info(f"Registered user: {name} ({roll_number}), {len(embeddings)} samples")

        return jsonify({
            "message": "Registration successful",
            "user": user.to_dict(),
            "samples_processed": len(embeddings),
            "embedding_dim": len(avg_embedding),
            **pipeline_result["images"],
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
