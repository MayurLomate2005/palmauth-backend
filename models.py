from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import json

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(50), unique=True, nullable=False)
    embedding = db.Column(db.Text, nullable=False)  # JSON-encoded list of floats
    num_samples = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    attendances = db.relationship("Attendance", backref="user", lazy=True)

    def get_embedding(self):
        return json.loads(self.embedding)

    def set_embedding(self, vec):
        self.embedding = json.dumps(vec)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "roll_number": self.roll_number,
            "num_samples": self.num_samples,
            "created_at": self.created_at.isoformat(),
        }


class Attendance(db.Model):
    __tablename__ = "attendance"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    date = db.Column(db.Date, default=date.today)
    time = db.Column(db.Time, default=lambda: datetime.utcnow().time())
    similarity_score = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default="success")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user.name if self.user else None,
            "roll_number": self.user.roll_number if self.user else None,
            "date": self.date.isoformat(),
            "time": self.time.strftime("%H:%M:%S"),
            "similarity_score": self.similarity_score,
            "status": self.status,
        }
