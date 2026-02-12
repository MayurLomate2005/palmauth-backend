from flask import Blueprint, request, jsonify, current_app
import cv2
import uuid
import logging
from datetime import date
from models import db, User, Attendance
from routes.auth import decode_image, process_pipeline
from services.matching import find_best_match

logger = logging.getLogger(__name__)

attendance_bp = Blueprint("attendance", __name__)


@attendance_bp.route("/api/authenticate", methods=["POST"])
def authenticate():
    """Authenticate user and mark attendance."""
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "Palm image is required"}), 400

        image = decode_image(image_data)
        uid = uuid.uuid4().hex[:12]
        config = current_app

        # Run pipeline
        result = process_pipeline(image, uid, config)
        if not result["success"]:
            return jsonify({
                "error": result.get("error", "Processing failed"),
                "status": "Failed",
            }), 400

        # Get all users
        users = User.query.all()
        if len(users) == 0:
            return jsonify({
                "error": "No registered users found",
                "status": "Failed",
                **result["images"],
            }), 404

        # Match
        threshold = current_app.config.get("SIMILARITY_THRESHOLD", 0.85)
        match = find_best_match(result["embedding"], users, threshold)

        response = {
            **result["images"],
            "similarity_score": match["similarity_score"],
            "threshold": match["threshold"],
            "status": match["status"],
        }

        if match["matched"]:
            user = User.query.get(match["user"]["id"])

            # Check duplicate attendance for today
            existing_attendance = Attendance.query.filter_by(
                user_id=user.id, date=date.today()
            ).first()

            if existing_attendance:
                response["message"] = f"Attendance already marked for {user.name} today"
                response["user"] = user.to_dict()
                response["duplicate"] = True
            else:
                # Mark attendance
                att = Attendance(
                    user_id=user.id,
                    similarity_score=match["similarity_score"],
                    status="success",
                )
                db.session.add(att)
                db.session.commit()

                response["message"] = f"Attendance marked for {user.name}"
                response["user"] = user.to_dict()
                response["attendance_id"] = att.id

            logger.info(f"Auth success: {user.name} (score={match['similarity_score']})")
        else:
            # Log failed attempt
            att = Attendance(
                user_id=0,
                similarity_score=match["similarity_score"],
                status="failed",
            )
            logger.info(f"Auth failed: score={match['similarity_score']}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Authentication error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@attendance_bp.route("/api/attendance", methods=["GET"])
def get_attendance():
    """Get attendance logs with optional date filter."""
    try:
        date_filter = request.args.get("date")
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 50, type=int)

        query = Attendance.query.filter(Attendance.status == "success")

        if date_filter:
            query = query.filter(Attendance.date == date_filter)

        query = query.order_by(Attendance.created_at.desc())
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        return jsonify({
            "attendance": [a.to_dict() for a in pagination.items],
            "total": pagination.total,
            "page": pagination.page,
            "pages": pagination.pages,
        }), 200

    except Exception as e:
        logger.error(f"Attendance fetch error: {str(e)}")
        return jsonify({"error": str(e)}), 500
