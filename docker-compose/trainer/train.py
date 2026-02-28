"""
FraudShield - Trainer Container
Saves all artifacts to shared Docker volume /artifacts
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

DATA_PATH = os.getenv("DATA_PATH", "/data/creditcard_sample.csv")
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "/artifacts")
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

print("=" * 50)
print("🚀 FraudShield Trainer Starting...")
print("=" * 50)

# Load
print("🔄 Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ {len(df):,} records | Fraud: {df['is_fraud'].mean()*100:.2f}%")

# Features
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

FEATURES = [
    'amt', 'hour', 'day_of_week', 'month',
    'distance', 'age', 'gender_enc', 'category_enc',
    'city_pop', 'zip'
]

X = df[FEATURES].fillna(0)
y = df['is_fraud']

# Split + Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
print(f"🔄 SMOTE... fraud before: {y_train.sum()}/{len(y_train)}")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
print(f"✅ After SMOTE: {sum(y_res==0):,} legit | {sum(y_res==1):,} fraud")

# Train
print("🔄 Training XGBoost...")
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
model.fit(X_res, y_res)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "total_test_samples": len(y_test),
    "fraud_detected": int(sum(y_pred)),
    "features": FEATURES
}

print("\n📊 Performance:")
for k, v in metrics.items():
    print(f"   {k}: {v}")

# Save to shared volume
with open(f"{ARTIFACTS_PATH}/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(f"{ARTIFACTS_PATH}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{ARTIFACTS_PATH}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(f"{ARTIFACTS_PATH}/category_map.json", "w") as f:
    json.dump(cat_map, f)

print(f"\n✅ All artifacts saved to {ARTIFACTS_PATH}")
print("🎉 Training complete!")