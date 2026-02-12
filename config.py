import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///palmauth.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    UPLOAD_ORIGINAL = os.path.join(UPLOAD_FOLDER, "original")
    UPLOAD_DETECTED = os.path.join(UPLOAD_FOLDER, "detected")
    UPLOAD_ROI = os.path.join(UPLOAD_FOLDER, "roi")
    UPLOAD_FEATURES = os.path.join(UPLOAD_FOLDER, "features")

    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload


# Ensure upload directories exist
for d in [
    Config.UPLOAD_ORIGINAL,
    Config.UPLOAD_DETECTED,
    Config.UPLOAD_ROI,
    Config.UPLOAD_FEATURES,
]:
    os.makedirs(d, exist_ok=True)
