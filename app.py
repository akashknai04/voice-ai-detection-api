from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
import base64
import librosa
import numpy as np
import tempfile
import os
import joblib
from datetime import datetime

app = FastAPI()

# 🔐 Production secret key
API_SECRET_KEY = "akash_secret_key"

# ✅ Load trained classifier
classifier = joblib.load("voice_classifier.pkl")

# 🛡️ Agentic Behavior Memory Store
agent_memory = {}

# ✅ Match GUVI camelCase field names
class AudioRequest(BaseModel):
    language: str
    audio_format: str = Field(alias="audioFormat")
    audio_base64_format: str = Field(alias="audioBase64")

    class Config:
        allow_population_by_field_name = True


@app.get("/health")
def health():
    return {"status": "ok"}


# 🔥 AI Voice Detection Endpoint
@app.post("/v1/detect")
def detect_audio(
    request: AudioRequest,
    request_obj: Request,
    x_api_key: str = Header(...)
):

    client_ip = request_obj.client.host
    now = datetime.utcnow()

    if client_ip not in agent_memory:
        agent_memory[client_ip] = {
            "risk_score": 0,
            "attempts": 0,
            "invalid_auth": 0,
            "last_seen": now
        }

    memory = agent_memory[client_ip]
    memory["attempts"] += 1
    memory["last_seen"] = now

    # 🔐 Authentication
    if x_api_key != API_SECRET_KEY:
        memory["invalid_auth"] += 1
        memory["risk_score"] += 2
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 🔹 Decode Base64
    try:
        audio_bytes = base64.b64decode(request.audio_base64_format)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 🔹 Save temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        y, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
        duration = len(y) / sr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    # 🔹 Feature Extraction
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    except Exception as e:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # 🔹 AI Prediction
    try:
        prediction = classifier.predict(mfcc_mean)[0]
        confidence = classifier.predict_proba(mfcc_mean)[0][prediction]
        voice_label = "ai_generated" if prediction == 1 else "human"
    except Exception as e:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    # 🧠 Risk Scoring
    session_risk = 3 if voice_label == "ai_generated" else 0
    memory["risk_score"] += session_risk
    total_risk = memory["risk_score"]

    if total_risk >= 8:
        threat_level = "CRITICAL"
    elif total_risk >= 5:
        threat_level = "HIGH"
    elif total_risk >= 3:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    os.remove(temp_audio_path)

    return {
        "status": "success",
        "result": {
            "is_fraud": threat_level in ["HIGH", "CRITICAL"],
            "threat_level": threat_level,
            "session_risk": session_risk,
            "total_behavior_risk": total_risk,
            "voice_classification": voice_label,
            "confidence": round(float(confidence), 3),
            "explanation": "Agentic AI fraud detection with behavior-based adaptive scoring."
        }
    }


# 🎯 Honeypot Endpoint
@app.post("/v1/honeypot")
def honeypot(request_obj: Request, x_api_key: str = Header(...)):

    client_ip = request_obj.client.host

    if client_ip not in agent_memory:
        agent_memory[client_ip] = {
            "risk_score": 0,
            "attempts": 0,
            "invalid_auth": 0
        }

    memory = agent_memory[client_ip]
    memory["attempts"] += 1

    if x_api_key != API_SECRET_KEY:
        memory["invalid_auth"] += 1
        memory["risk_score"] += 3
    else:
        memory["risk_score"] += 1

    total_risk = memory["risk_score"]

    if total_risk >= 8:
        threat_level = "CRITICAL"
    elif total_risk >= 5:
        threat_level = "HIGH"
    elif total_risk >= 3:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    return {
        "status": "monitored",
        "threat_level": threat_level,
        "behavior_risk_score": total_risk,
        "total_requests": memory["attempts"]
    }
