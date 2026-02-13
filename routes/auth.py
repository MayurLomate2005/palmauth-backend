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


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def decode_image(data: str) -> np.ndarray:
    """Decode base64 image to OpenCV format."""
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def image_to_base64(path: str) -> str:
    """Convert image file to base64 string."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def process_pipeline(image: np.ndarray, uid: str, app) -> dict:
    """Run full palm processing pipeline."""
    config = app.config

    # Step 1: Save original
    orig_path = os.path.join(config["UPLOAD_ORIGINAL"], f"{uid}.jpg")
    cv2.imwrite(orig_path, image)

    # Step 2: Palm detection
    det_path = os.path.join(config["UPLOAD_DETECTED"], f"{uid}.jpg")
    detection = detect_palm(image, det_path)

    if not detection["success"]:
        return {"success": False, "error": detection["error"]}

    # Step 3: ROI extraction
    roi_path = os.path.join(config["UPLOAD_ROI"], f"{uid}.jpg")
    roi_result = extract_roi(image, detection["bbox"], roi_path)

    if not roi_result["success"]:
        return {"success": False, "error": roi_result["error"]}

    # Step 4: Feature visualization
    feat_path = os.path.join(config["UPLOAD_FEATURES"], f"{uid}.jpg")
    feat_result = visualize_features(roi_path, feat_path)

    # Step 5: Embedding extraction
    emb_result = extract_embedding(
        roi_result["roi_image"], 
        config["EMBEDDING_DIM"]
    )

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


# --------------------------------------------------
# REGISTER ROUTE
# --------------------------------------------------

@auth_bp.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        name = data.get("name")
        roll_number = data.get("roll_number")
        images = data.get("images", [])

        if not name or not roll_number:
            return jsonify({"error": "Name and roll_number are required"}), 400

        if not images:
            return jsonify({"error": "At least one palm image is required"}), 400

        # Check duplicate user
        existing = User.query.filter_by(roll_number=roll_number).first()
        if existing:
            return jsonify({"error": "User already exists"}), 409

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

        if not embeddings:
            return jsonify({"error": "No valid palm images processed"}), 400

        # Average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)

        if norm > 0:
            avg_embedding = avg_embedding / norm

        user = User(
            name=name,
            roll_number=roll_number,
            num_samples=len(embeddings),
        )

        user.set_embedding(avg_embedding.tolist())

        db.session.add(user)
        db.session.commit()

        logger.info(f"Registered user: {name}")

        return jsonify({
            "success": True,
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


# --------------------------------------------------
# AUTHENTICATE ROUTE
# --------------------------------------------------

@auth_bp.route("/api/authenticate", methods=["POST"])
def authenticate():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "Palm image is required"}), 400

        image = decode_image(image_data)
        uid = f"auth_{uuid.uuid4().hex[:8]}"

        result = process_pipeline(image, uid, current_app)

        if not result["success"]:
            return jsonify({"error": result["error"]}), 400

        input_embedding = np.array(result["embedding"])

        users = User.query.all()

        if not users:
            return jsonify({"error": "No registered users found"}), 404

        best_user = None
        best_similarity = -1

        for user in users:
            stored_embedding = np.array(user.get_embedding())
            similarity = float(np.dot(input_embedding, stored_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_user = user

        threshold = 0.85

        if best_similarity >= threshold:
            return jsonify({
                "success": True,
                "matched": True,
                "confidence": best_similarity,
                "user": best_user.to_dict(),
                **result["images"]
            }), 200

        return jsonify({
            "success": True,
            "matched": False,
            "confidence": best_similarity,
            "message": "No matching user found"
        }), 200

    except Exception as e:
        logger.error(f"Authentication error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
