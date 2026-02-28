# 🐳 FraudShield — Multi-Service Docker Compose

A full-stack fraud detection platform orchestrating 4 Docker containers — FastAPI, Streamlit, PostgreSQL, and a model trainer — via Docker Compose.

---

## 🎯 What This Does

Orchestrates 4 services that work together to provide a complete fraud detection platform with real-time predictions, a live dashboard, and persistent prediction logging.

---

## 🏗️ Architecture

```
docker-compose.yml
├── trainer    → Trains XGBoost model → saves to shared volume
├── postgres   → Stores all predictions (PostgreSQL 15)
├── ml-api     → FastAPI serving predictions + logging to DB
└── dashboard  → Streamlit live UI consuming the API
```

---

## 🗂️ Files

```
docker-compose/
├── docker-compose.yml       → Service orchestration
├── .env                     → Environment variables
├── ml-api/
│   ├── app.py               → FastAPI + PostgreSQL logging
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── streamlit_app.py     → Live fraud detection UI
│   ├── Dockerfile
│   └── requirements.txt
├── trainer/
│   ├── train.py             → XGBoost + SMOTE training
│   ├── Dockerfile
│   └── requirements.txt
└── postgres/
    └── init.sql             → DB schema + fraud summary view
```

---

## 🔌 Ports

| Service | Port | URL |
|---------|------|-----|
| ml-api | 8001 | http://localhost:8001 |
| dashboard | 8501 | http://localhost:8501 |
| postgres | 5432 | Internal only |

---

## 🚀 How to Run

### Step 1 — Prepare Data
```bash
# Place your dataset
mkdir -p data
cp /path/to/creditcard_sample.csv ./data/
```

### Step 2 — Train Model
```bash
docker compose run --rm trainer
```

Expected output:
```
✅ 58,694 records | Fraud: 0.98%
✅ After SMOTE: 46,495 legit | 46,495 fraud
📊 roc_auc: 0.9947
✅ All artifacts saved to /artifacts
🎉 Training complete!
```

### Step 3 — Start All Services
```bash
docker compose up -d
```

### Step 4 — Verify All Running
```bash
docker compose ps
```

Expected:
```
NAME                    STATUS
fraudshield-api-v2      Up (healthy)
fraudshield-dashboard   Up (healthy)
fraudshield-postgres    Up (healthy)
```

### Step 5 — Test
```bash
# API health
curl http://localhost:8001/health

# Open dashboard
open http://localhost:8501
```

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health + DB status |
| GET | `/metrics` | Model + runtime stats |
| GET | `/history` | Prediction history from PostgreSQL |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

---

## 📊 Streamlit Dashboard Pages

| Page | Description |
|------|-------------|
| 🔍 Predict | Input form → fraud/legitimate result + risk gauge |
| 📊 Dashboard | Live charts — fraud rate, amount distribution |
| 📜 History | All predictions from PostgreSQL |
| ℹ️ Model Info | Performance metrics, feature list |

---

## 🗄️ PostgreSQL

Connect directly:
```bash
docker exec -it fraudshield-postgres psql -U frauduser -d fraudshield
```

Useful queries:
```sql
-- All predictions
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;

-- Fraud summary
SELECT * FROM fraud_summary;

-- Fraud rate
SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction;
```

---

## ⚙️ Environment Variables (.env)

```
POSTGRES_DB=fraudshield
POSTGRES_USER=frauduser
POSTGRES_PASSWORD=fraudpass123
```

---

## 🐳 Docker Compose Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f ml-api

# Rebuild specific service
docker compose up -d --build dashboard

# Remove everything including volumes
docker compose down -v

# Check status
docker compose ps
```

---

## ☁️ GCP Deployment

```bash
# On GCP VM
sudo docker compose run --rm trainer
sudo docker compose up -d
sudo docker compose ps

# Access via public IP
curl http://<VM_EXTERNAL_IP>:8001/health
# Open http://<VM_EXTERNAL_IP>:8501
```

---

## 📸 Screenshots

**All services healthy:**
<img width="1194" height="81" alt="image" src="https://github.com/user-attachments/assets/e656e663-3297-4d7a-b8f5-5338a2182f12" />


**Streamlit — FRAUD prediction:**
<img width="1639" height="930" alt="image" src="https://github.com/user-attachments/assets/6fd523b3-3d2b-4cd1-90ce-b9680feac319" />


**Streamlit — Live Dashboard:**
<img width="1689" height="888" alt="image" src="https://github.com/user-attachments/assets/5f5134d9-878f-4738-890e-918c053e5017" />


**Streamlit — Prediction History:**
<img width="1658" height="596" alt="image" src="https://github.com/user-attachments/assets/109a6da2-5e5d-4308-a998-86e1c7ea08e9" />


**PostgreSQL predictions:**
<img width="1196" height="214" alt="image" src="https://github.com/user-attachments/assets/8b7921ce-e25a-4d3a-b762-173ee680e7a9" />


**GCP Deployment:**
<img width="1498" height="124" alt="image" src="https://github.com/user-attachments/assets/8584f52b-d186-40e3-aec4-9f5c64badc26" />

