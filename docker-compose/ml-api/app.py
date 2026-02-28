"""
FraudShield - FastAPI Prediction Service
Lab 2: Multi-container — WITH PostgreSQL logging
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import json
import numpy as np
from datetime import datetime
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import time

# ─────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────
app = FastAPI(
    title="FraudShield ML API",
    description="Credit Card Fraud Detection — Docker Compose Lab",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load Artifacts from shared volume
# ─────────────────────────────────────────────
with open("/artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("/artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("/artifacts/metrics.json", "r") as f:
    model_metrics = json.load(f)

with open("/artifacts/category_map.json", "r") as f:
    category_map = json.load(f)

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
def get_db():
    retries = 5
    while retries > 0:
        try:
            return psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "postgres"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "fraudshield"),
                user=os.getenv("POSTGRES_USER", "frauduser"),
                password=os.getenv("POSTGRES_PASSWORD", "fraudpass123")
            )
        except:
            retries -= 1
            time.sleep(2)
    raise Exception("Cannot connect to PostgreSQL")

def log_prediction(tx_id, amount, category, prediction, probability, confidence):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions
            (transaction_id, amount, category, prediction, fraud_probability, confidence, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (tx_id, amount, category, prediction, probability, confidence, datetime.utcnow()))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ DB log failed: {e}")

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class TransactionInput(BaseModel):
    amt: float = Field(..., example=149.99)
    hour: int = Field(..., example=14)
    day_of_week: int = Field(..., example=2)
    month: int = Field(..., example=6)
    distance: float = Field(..., example=12.5)
    age: int = Field(..., example=35)
    gender: str = Field(..., example="M")
    category: str = Field(..., example="shopping_net")
    city_pop: int = Field(..., example=50000)
    zip: int = Field(..., example=12345)

class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: str
    fraud_probability: float
    confidence: str
    timestamp: str
    amount: float
    category: str

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "FraudShield ML API v2",
        "status": "running",
        "endpoints": ["/predict", "/health", "/metrics", "/history", "/docs"]
    }

@app.get("/health")
def health():
    db_status = "unknown"
    try:
        conn = get_db()
        conn.close()
        db_status = "connected"
    except:
        db_status = "disconnected"
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
def get_metrics():
    total, fraud_count = 0, 0
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'FRAUD'")
        fraud_count = cur.fetchone()[0]
        cur.close()
        conn.close()
    except:
        pass
    return {
        "model": "XGBoost + SMOTE",
        "dataset": "Kaggle Credit Card Transactions",
        "performance": model_metrics,
        "runtime_stats": {
            "total_predictions": total,
            "fraud_detected": fraud_count,
            "fraud_rate": round(fraud_count / total * 100, 2) if total > 0 else 0
        }
    }

@app.get("/history")
def get_history(limit: int = 50):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"predictions": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        gender_enc = 1 if transaction.gender.upper() == "M" else 0
        category_enc = category_map.get(transaction.category, 0)

        features = np.array([[
            transaction.amt, transaction.hour, transaction.day_of_week,
            transaction.month, transaction.distance, transaction.age,
            gender_enc, category_enc, transaction.city_pop, transaction.zip
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        confidence = "HIGH" if probability > 0.85 else "MEDIUM" if probability > 0.5 else "LOW"
        tx_id = f"TX-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:18]}"
        pred_label = "FRAUD" if prediction == 1 else "LEGITIMATE"

        log_prediction(tx_id, transaction.amt, transaction.category,
                      pred_label, float(probability), confidence)

        return PredictionResponse(
            transaction_id=tx_id,
            prediction=pred_label,
            fraud_probability=round(float(probability), 4),
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            amount=transaction.amt,
            category=transaction.category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionInput]):
    return [predict(t) for t in transactions]