import os
import logging
from flask import Flask, send_from_directory
from flask_cors import CORS
from config import Config
from models import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("palmauth.log"),
    ],
)
logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Fix Render PostgreSQL URL (postgres:// -> postgresql://)
    uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if uri.startswith("postgres://"):
        app.config["SQLALCHEMY_DATABASE_URI"] = uri.replace(
            "postgres://", "postgresql://", 1
        )

    # CORS
    CORS(
        app,
        origins=[Config.FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Database
    db.init_app(app)

    with app.app_context():
        db.create_all()
        logger.info("Database tables created")

    # Register blueprints
    from routes.auth import auth_bp
    from routes.attendance import attendance_bp
    from routes.users import users_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(users_bp)

    # Serve uploaded images
    @app.route("/uploads/<path:filename>")
    def serve_upload(filename):
        return send_from_directory(Config.UPLOAD_FOLDER, filename)

    # Health check
    @app.route("/api/health")
    def health():
        return {"status": "healthy", "version": "1.0.0"}

    logger.info("PalmAuth backend initialized")
    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") != "production")
