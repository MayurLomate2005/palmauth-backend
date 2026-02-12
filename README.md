# PalmAuth Backend

Production-ready Flask backend for the Contactless Palmprint Biometric Authentication System.

## Tech Stack
- Python 3.10+
- Flask + Flask-CORS
- PyTorch (ResNet18 for feature extraction)
- MediaPipe Hands (palm detection)
- OpenCV (image preprocessing)
- PostgreSQL (database)
- Gunicorn (production server)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Initialize database
python -c "from app import db; db.create_all()"

# Run development server
python app.py

# Run production server
gunicorn app:app --bind 0.0.0.0:5000 --workers 4
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/register` | Register new user with palm images |
| POST | `/api/authenticate` | Authenticate user and mark attendance |
| GET | `/api/attendance` | Get attendance logs |
| GET | `/api/metrics` | Get system performance metrics |
| GET | `/api/users` | List all registered users |

## Deployment (Render)

1. Push this backend folder to a separate GitHub repo
2. Create a new Web Service on Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 4`
5. Add environment variables from `.env.example`
6. Add a PostgreSQL database from Render dashboard

## Project Structure

```
backend/
├── app.py                 # Main Flask application
├── config.py              # Configuration
├── models.py              # Database models
├── requirements.txt       # Dependencies
├── render.yaml            # Render deployment config
├── .env.example           # Environment template
├── routes/
│   ├── auth.py            # Authentication endpoints
│   ├── attendance.py      # Attendance endpoints
│   └── users.py           # User management endpoints
├── services/
│   ├── palm_detection.py  # MediaPipe palm detection
│   ├── roi_extraction.py  # ROI cropping & preprocessing
│   ├── feature_extraction.py  # CNN embedding extraction
│   ├── feature_visualization.py  # Edge/line detection overlay
│   └── matching.py        # Cosine similarity matching
└── uploads/
    ├── original/
    ├── detected/
    ├── roi/
    └── features/
```
