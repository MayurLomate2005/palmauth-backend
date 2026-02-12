import os
import logging
from flask import Flask, send_from_directory
from flask_cors import CORS
from config import Config
from models import db

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # -------------------------
    # Fix Render PostgreSQL URL
    # -------------------------
    uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if uri.startswith("postgres://"):
        app.config["SQLALCHEMY_DATABASE_URI"] = uri.replace(
            "postgres://", "postgresql://", 1
        )

    # -------------------------
    # Ensure upload folder exists
    # -------------------------
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    # -------------------------
    # Enable CORS
    # -------------------------
    CORS(
        app,
        origins=[
            Config.FRONTEND_URL,
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        supports_credentials=True,
    )

    # -------------------------
    # Initialize Database
    # -------------------------
    db.init_app(app)

    with app.app_context():
        db.create_all()
        logger.info("Database tables created")

    # -------------------------
    # Register Blueprints
    # -------------------------
    from routes.auth import auth_bp
    from routes.attendance import attendance_bp
    from routes.users import users_bp

    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(attendance_bp, url_prefix="/api")
    app.register_blueprint(users_bp, url_prefix="/api")

    # -------------------------
    # Root Route (Fix 404)
    # -------------------------
    @app.route("/")
    def home():
        return {
            "message": "PalmAuth Backend Running 🚀",
            "status": "ok",
            "version": "1.0.0"
        }

    # -------------------------
    # Health Check Route
    # -------------------------
    @app.route("/api/health")
    def health():
        return {
            "status": "healthy",
            "service": "PalmAuth Backend",
            "version": "1.0.0"
        }

    # -------------------------
    # Serve Uploaded Images
    # -------------------------
    @app.route("/uploads/<path:filename>")
    def serve_upload(filename):
        return send_from_directory(Config.UPLOAD_FOLDER, filename)

    logger.info("PalmAuth backend initialized")
    return app


# Create app instance
app = create_app()

# -------------------------
# Local Development Run
# -------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
