# 🛡️ FraudShield — ML Fraud Detection Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?style=flat-square&logo=postgresql)
![GCP](https://img.shields.io/badge/GCP-Deployed-4285F4?style=flat-square&logo=googlecloud)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange?style=flat-square)

> **Author:** Nagashree Bommenahalli Kumaraswamy  
> **Course:** DADS 7305 — MLOps | Northeastern University  
> **Live on GCP:** `http://136.114.127.5:8001` | `http://136.114.127.5:8501`

---

## 🎯 What It Does

FraudShield is a **production-grade, fully containerized credit card fraud detection system** that:

- 🧠 Trains an **XGBoost model with SMOTE** on real transaction data
- ⚡ Serves real-time fraud predictions via a **FastAPI REST API**
- 📊 Provides a **live Streamlit dashboard** for fraud analysis
- 🗄️ Logs every prediction to **PostgreSQL** for audit trails
- ☁️ Deployed on **Google Cloud Platform** VM

---

## 🏗️ Architecture

```
GCP VM (Ubuntu 24.04 | e2-medium | us-central1-a)
└── docker compose up
    ├── 🔄 trainer      → XGBoost + SMOTE → shared volume
    ├── 🗄️  postgres     → prediction logging
    ├── 🧠 ml-api       → FastAPI REST API  :8001
    └── 📊 dashboard    → Streamlit UI      :8501
```

---

## 🔬 Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.9947** |
| Precision | 0.8673 |
| Recall | 0.8522 |
| F1 Score | 0.8596 |

> XGBoost Classifier trained on Kaggle Credit Card Transactions dataset with SMOTE oversampling to handle 0.98% fraud rate imbalance.

---

## ☁️ GCP Deployment

| Service | URL |
|---------|-----|
| 🧠 REST API | http://136.114.127.5:8001 |
| 📖 API Docs | http://136.114.127.5:8001/docs |
| 📊 Dashboard | http://136.114.127.5:8501 |

### GCP VM — All Services Running
<img width="1503" height="248" alt="image" src="https://github.com/user-attachments/assets/762070c6-4be6-4a8b-afaa-530a2a554b27" />


### API Live on Public IP
<img width="829" height="182" alt="image" src="https://github.com/user-attachments/assets/3c2b4661-6e19-4527-b6d7-411d6e3c7328" />


### Streamlit Dashboard on GCP
<img width="1563" height="994" alt="image" src="https://github.com/user-attachments/assets/c685a23c-5e1b-41df-bf27-98802d0e15ed" />

---

## 📊 Streamlit Dashboard

### Real-Time Fraud Detection
<img width="1604" height="862" alt="image" src="https://github.com/user-attachments/assets/70ea2e53-d667-4ef7-b1a7-b9a1052a4e5f" />


### Live Analytics Dashboard
<img width="1615" height="917" alt="image" src="https://github.com/user-attachments/assets/e6394ce3-b6f7-4a3e-b95f-8b7a2503c4bf" />


### Prediction History from PostgreSQL
<img width="1636" height="752" alt="image" src="https://github.com/user-attachments/assets/6bb08013-a4c4-4f8c-894e-b80dc50f8a71" />


---

## 🧪 API Testing (Postman)

### FRAUD Detection — 93.26% probability
<img width="1533" height="720" alt="image" src="https://github.com/user-attachments/assets/f5e26a34-ccae-484c-9bb5-673d862f40a7" />


### LEGITIMATE Transaction
<img width="1551" height="814" alt="image" src="https://github.com/user-attachments/assets/d59d2010-b456-4133-856a-8db6d87d32a8" />


---

## 🐳 Docker

### Single Container (docker-basics)
<img width="1088" height="196" alt="image" src="https://github.com/user-attachments/assets/324a80db-3956-4831-9687-204a78926869" />


### Multi-Service (docker-compose) — All Healthy
<img width="1199" height="147" alt="image" src="https://github.com/user-attachments/assets/f3be1b89-2c9e-426f-b4ad-63460060cca7" />


---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/NagashreeBK98/fraudshield-mlops.git

# Single container
cd docker-basics
python train.py
docker build -t fraudshield-api:v1 .
docker run -d -p 8000:8000 --name fraudshield-api fraudshield-api:v1

# Multi-service
cd docker-compose
docker compose run --rm trainer
docker compose up -d
```

---

## 📁 Project Structure

```
fraudshield-mlops/
├── docker-basics/          ← Single container
│   ├── train.py
│   ├── app.py
│   └── Dockerfile
├── docker-compose/         ← Multi-service
│   ├── docker-compose.yml
│   ├── ml-api/
│   ├── dashboard/
│   ├── trainer/
│   └── postgres/
├── postman/

```

---

## 🔑 MLOps Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Containerization | Multi-stage Dockerfiles |
| Orchestration | Docker Compose — 4 services |
| Shared Volumes | Model artifacts between services |
| Health Checks | All services monitored |
| Config Management | `.env` secrets |
| Persistent Storage | PostgreSQL named volumes |
| REST API | FastAPI + Pydantic validation |
| Class Imbalance | SMOTE oversampling |
| Cloud Deployment | GCP Compute Engine |
| Real-time UI | Streamlit dashboard |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost, scikit-learn, imbalanced-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Database | PostgreSQL 15 |
| Containerization | Docker, Docker Compose |
| Cloud | Google Cloud Platform (Compute Engine) |
| Language | Python 3.10 |

---

## 📬 API Endpoints

```bash
GET  /health          # Health + DB status
GET  /metrics         # Model performance + runtime stats
GET  /history         # Prediction history from PostgreSQL
POST /predict         # Single transaction prediction
POST /predict/batch   # Batch predictions
GET  /docs            # Interactive Swagger UI
```

---

## 📊 Sample Request & Response

```json
POST /predict
{
  "amt": 4999.99,
  "hour": 3,
  "day_of_week": 6,
  "month": 12,
  "distance": 180.5,
  "age": 22,
  "gender": "M",
  "category": "shopping_net",
  "city_pop": 800,
  "zip": 501
}
```

```json
{
  "transaction_id": "TX-20260228011751...",
  "prediction": "FRAUD",
  "fraud_probability": 0.9326,
  "confidence": "HIGH"
}
```

---

## 🔗 References

- [Prof. Ramin Mohammadi MLOps Repo](https://github.com/raminmohammadi/MLOps)
- [Kaggle Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
