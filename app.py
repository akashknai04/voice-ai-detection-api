from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import tempfile
import os
import joblib

app = FastAPI()

# 🔐 Use environment variable in production
API_SECRET_KEY = "akash_secret_key"

# ✅ Load trained classifier (lightweight)
classifier = joblib.load("voice_classifier.pkl")


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
    x_api_key: str = Header(...)
):
    # 🔐 API Key validation
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized access attempt logged.")

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
        os.remove(temp_audio_path)

        duration = len(y) / sr

        if duration < 0.5:
            raise HTTPException(status_code=400, detail="Audio too short")

        if duration > 15:
            raise HTTPException(status_code=400, detail="Audio too long")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    # 🔹 Extract MFCC Features (LIGHTWEIGHT 🔥)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # 🔹 Prediction
    try:
        prediction = classifier.predict(mfcc_mean)[0]
        confidence = classifier.predict_proba(mfcc_mean)[0][prediction]

        label = "ai_generated" if prediction == 1 else "human"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    return {
        "classification": label,
        "confidence": round(float(confidence), 3),
        "language": request.language,
        "meta": {
            "sample_rate": sr,
            "duration_seconds": round(duration, 2)
        },
        "security": {
            "authenticated": True,
            "honeypot_flag": False
        }
    }
