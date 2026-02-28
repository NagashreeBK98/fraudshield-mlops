"""
FraudShield - Model Training Script
Dataset: Kaggle Credit Card Transactions Dataset
Model: XGBoost with SMOTE for class imbalance
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "../data/creditcard_sample.csv")

print("🔄 Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df):,} records | Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────
print("🔄 Engineering features...")

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['month'] = df['trans_date_trans_time'].dt.month

df['distance'] = np.sqrt(
    (df['lat'] - df['merch_lat'])**2 +
    (df['long'] - df['merch_long'])**2
)

df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365
df['gender_enc'] = (df['gender'] == 'M').astype(int)

cat_map = {c: i for i, c in enumerate(df['category'].unique())}
df['category_enc'] = df['category'].map(cat_map)

# Save category map for inference
with open("category_map.json", "w") as f:
    json.dump(cat_map, f)

FEATURES = [
    'amt', 'hour', 'day_of_week', 'month',
    'distance', 'age', 'gender_enc', 'category_enc',
    'city_pop', 'zip'
]

X = df[FEATURES].fillna(0)
y = df['is_fraud']

# ─────────────────────────────────────────────
# 3. Train/Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 4. Scale Features
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. SMOTE — Handle Class Imbalance
# ─────────────────────────────────────────────
print("🔄 Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print(f"✅ After SMOTE: {sum(y_train_res==0):,} legit | {sum(y_train_res==1):,} fraud")

# ─────────────────────────────────────────────
# 6. Train XGBoost
# ─────────────────────────────────────────────
print("🔄 Training XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# ─────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "roc_auc":  round(roc_auc_score(y_test, y_prob), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall":   round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "total_test_samples": len(y_test),
    "fraud_detected": int(sum(y_pred)),
    "features": FEATURES
}

print("\n📊 Model Performance:")
for k, v in metrics.items():
    print(f"   {k}: {v}")
print("\n", classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# ─────────────────────────────────────────────
# 8. Save Artifacts
# ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Saved: model.pkl | scaler.pkl | metrics.json | category_map.json")
print("🎉 Training complete!")