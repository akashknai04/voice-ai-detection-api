from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import tempfile
import os
import joblib
import datetime

app = FastAPI()

# 🔐 Use environment variable in production
API_SECRET_KEY = "akash_secret_key"

# ✅ Load trained classifier (lightweight)
classifier = joblib.load("voice_classifier.pkl")

# 🛡️ In-memory request behavior tracker
request_tracker = {}

# 🔥 Fraud keyword list (future extension ready)
FRAUD_KEYWORDS = [
    "otp", "bank", "account", "verify", "urgent",
    "transfer", "credit card", "debit card",
    "password", "pin", "aadhaar", "pan",
    "reward", "lottery", "investment"
]


class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_b64_mp3: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/detect")
def detect_audio(
    request: AudioRequest,
    request_obj: Request,
    x_api_key: str = Header(...)
):

    # 🔐 Honeypot API Key validation
    if x_api_key != API_SECRET_KEY:
        print("⚠ Unauthorized access attempt detected")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized access attempt logged."
        )

    # 🛡️ Capture Client IP
    client_ip = request_obj.client.host
    current_time = datetime.datetime.utcnow()

    if client_ip not in request_tracker:
        request_tracker[client_ip] = {
            "count": 0,
            "last_request": current_time
        }

    # Reset if 60 seconds passed
    time_diff = (current_time - request_tracker[client_ip]["last_request"]).seconds

    if time_diff > 60:
        request_tracker[client_ip]["count"] = 0

    request_tracker[client_ip]["count"] += 1
    request_tracker[client_ip]["last_request"] = current_time

    # 🔹 Decode Base64
    try:
        audio_bytes = base64.b64decode(request.audio_b64_mp3)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 🔹 Load audio
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        y, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
        duration = len(y) / sr

        if duration < 0.5:
            os.remove(temp_audio_path)
            raise HTTPException(status_code=400, detail="Audio too short")

        if duration > 15:
            os.remove(temp_audio_path)
            raise HTTPException(status_code=400, detail="Audio too long")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    # 🔹 Extract MFCC Features
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

    except Exception as e:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # 🔹 AI Voice Classification
    try:
        prediction = classifier.predict(mfcc_mean)[0]
        confidence = classifier.predict_proba(mfcc_mean)[0][prediction]
        voice_label = "ai_generated" if prediction == 1 else "human"

    except Exception as e:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    # 🛡️ Intelligent Risk Scoring Engine
    risk_score = 0

    # AI voice increases risk
    if voice_label == "ai_generated":
        risk_score += 3

    # Too many requests from same IP
    if request_tracker[client_ip]["count"] > 5:
        risk_score += 2

    # Very short audio suspicious
    if duration < 1:
        risk_score += 1

    # Threat level classification
    if risk_score >= 5:
        threat_level = "HIGH"
    elif risk_score >= 3:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    honeypot_flag = threat_level == "HIGH"

    # Clean temp file
    os.remove(temp_audio_path)

    return {
        "status": "success",
        "result": {
            "is_fraud": honeypot_flag,
            "risk_score": risk_score,
            "threat_level": threat_level,
            "voice_classification": voice_label,
            "confidence": round(float(confidence), 3),
            "explanation": "AI voice fraud detection with intelligent behavior scoring."
        },
        "audio_metadata": {
            "language": request.language,
            "sample_rate": sr,
            "duration_seconds": round(duration, 2)
        },
        "security": {
            "authenticated": True,
            "honeypot_flag": honeypot_flag,
            "request_count_last_minute": request_tracker[client_ip]["count"]
        }
    }
