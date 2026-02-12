from flask import Blueprint, jsonify
from datetime import date, timedelta
from sqlalchemy import func
from models import db, User, Attendance

users_bp = Blueprint("users", __name__)


@users_bp.route("/api/users", methods=["GET"])
def get_users():
    """List all registered users."""
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({"users": [u.to_dict() for u in users]}), 200


@users_bp.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Get system performance metrics."""
    total_users = User.query.count()
    today = date.today()

    # Today's attendance
    today_attendance = Attendance.query.filter(
        Attendance.date == today, Attendance.status == "success"
    ).count()

    # Total authentications
    total_auths = Attendance.query.count()
    successful_auths = Attendance.query.filter_by(status="success").count()
    failed_auths = Attendance.query.filter_by(status="failed").count()

    # Accuracy
    accuracy = (successful_auths / total_auths * 100) if total_auths > 0 else 0

    # Average similarity for successful matches
    avg_similarity = (
        db.session.query(func.avg(Attendance.similarity_score))
        .filter(Attendance.status == "success")
        .scalar()
        or 0
    )

    # Weekly attendance data
    weekly = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        count = Attendance.query.filter(
            Attendance.date == d, Attendance.status == "success"
        ).count()
        weekly.append({"date": d.isoformat(), "count": count})

    # Similarity distribution
    ranges = [
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
        (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.01),
    ]
    similarity_dist = []
    for low, high in ranges:
        count = Attendance.query.filter(
            Attendance.similarity_score >= low,
            Attendance.similarity_score < high,
        ).count()
        similarity_dist.append({
            "range": f"{low:.2f}-{high:.2f}",
            "count": count,
        })

    return jsonify({
        "total_users": total_users,
        "today_attendance": today_attendance,
        "total_authentications": total_auths,
        "successful_authentications": successful_auths,
        "failed_authentications": failed_auths,
        "accuracy": round(accuracy, 2),
        "average_similarity": round(float(avg_similarity), 4),
        "false_rejection_count": failed_auths,
        "far": round((failed_auths / total_auths * 100) if total_auths > 0 else 0, 2),
        "frr": round((failed_auths / total_auths * 100) if total_auths > 0 else 0, 2),
        "weekly_attendance": weekly,
        "similarity_distribution": similarity_dist,
    }), 200
