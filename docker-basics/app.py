"""
FraudShield - FastAPI Prediction Service
Lab 1: Single Container — NO database
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import json
import numpy as np
from datetime import datetime
import os

# ─────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────
app = FastAPI(
    title="FraudShield ML API",
    description="Credit Card Fraud Detection — Docker Basics Lab",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load Artifacts
# ─────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("metrics.json", "r") as f:
    model_metrics = json.load(f)

with open("category_map.json", "r") as f:
    category_map = json.load(f)

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
        "service": "FraudShield ML API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/predict", "/health", "/metrics", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
def get_metrics():
    return {
        "model": "XGBoost + SMOTE",
        "dataset": "Kaggle Credit Card Transactions",
        "performance": model_metrics
    }

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

        return PredictionResponse(
            transaction_id=tx_id,
            prediction="FRAUD" if prediction == 1 else "LEGITIMATE",
            fraud_probability=round(float(probability), 4),
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            amount=transaction.amt,
            category=transaction.category
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchInput(BaseModel):
    transactions: list[TransactionInput]

@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    return [predict(t) for t in batch.transactions]