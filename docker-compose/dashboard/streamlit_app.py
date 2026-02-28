"""
FraudShield - Streamlit Dashboard
Real-time fraud detection UI
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

import os
API_URL = os.getenv("API_URL", "http://ml-api:8000")

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.fraud-box {
    background-color: #ff4b4b22;
    border-left: 5px solid #ff4b4b;
    padding: 1rem; border-radius: 0.5rem;
}
.legit-box {
    background-color: #00c85322;
    border-left: 5px solid #00c853;
    padding: 1rem; border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.title("🛡️ FraudShield")
    st.caption("MLOps Docker Lab")
    st.divider()
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("🟢 API Online")
        st.caption(f"DB: {h.get('database','unknown')}")
    except:
        st.error("🔴 API Offline")
    st.divider()
    page = st.radio("Navigate", [
        "🔍 Predict",
        "📊 Dashboard",
        "📜 History",
        "ℹ️ Model Info"
    ])

# ── Page: Predict ─────────────────────────────
if page == "🔍 Predict":
    st.title("🔍 Real-Time Fraud Detection")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amt = st.number_input("Amount ($)", min_value=0.01, value=149.99)
            category = st.selectbox("Category", [
                "shopping_net","grocery_pos","entertainment",
                "gas_transport","food_dining","health_fitness",
                "shopping_pos","misc_net","travel","personal_care"
            ])
            gender = st.selectbox("Gender", ["M","F"])
        with c2:
            hour = st.slider("Hour", 0, 23, 14)
            dow = st.selectbox("Day", [0,1,2,3,4,5,6],
                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
            month = st.selectbox("Month", list(range(1,13)))
        with c3:
            age = st.number_input("Age", 18, 100, 35)
            distance = st.number_input("Distance", 0.0, value=12.5)
            city_pop = st.number_input("City Population", 100, value=50000)
            zip_code = st.number_input("ZIP", 10000, 99999, 12345)

        submitted = st.form_submit_button("🚀 Analyze", use_container_width=True)

    if submitted:
        payload = {
            "amt": amt, "hour": hour, "day_of_week": dow,
            "month": month, "distance": distance, "age": age,
            "gender": gender, "category": category,
            "city_pop": city_pop, "zip": zip_code
        }
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        c1, c2 = st.columns([2,1])
        with c1:
            if res["prediction"] == "FRAUD":
                st.markdown(f"""<div class="fraud-box">
                    <h2>🚨 FRAUD DETECTED</h2>
                    <p>ID: <b>{res['transaction_id']}</b></p>
                    <p>Amount: <b>${res['amount']:.2f}</b> | Category: <b>{res['category']}</b></p>
                    <p>Fraud Probability: <b>{res['fraud_probability']*100:.1f}%</b> | Confidence: <b>{res['confidence']}</b></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="legit-box">
                    <h2>✅ LEGITIMATE</h2>
                    <p>ID: <b>{res['transaction_id']}</b></p>
                    <p>Amount: <b>${res['amount']:.2f}</b> | Category: <b>{res['category']}</b></p>
                    <p>Fraud Probability: <b>{res['fraud_probability']*100:.1f}%</b> | Confidence: <b>{res['confidence']}</b></p>
                </div>""", unsafe_allow_html=True)

        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["fraud_probability"]*100,
                title={"text": "Fraud Risk %"},
                gauge={
                    "axis": {"range": [0,100]},
                    "bar": {"color": "red" if res["prediction"]=="FRAUD" else "green"},
                    "steps": [
                        {"range": [0,40], "color": "lightgreen"},
                        {"range": [40,70], "color": "lightyellow"},
                        {"range": [70,100], "color": "lightsalmon"}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

# ── Page: Dashboard ───────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 Live Dashboard")

    try:
        metrics = requests.get(f"{API_URL}/metrics", timeout=3).json()
        rs = metrics.get("runtime_stats", {})
        perf = metrics.get("performance", {})

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Predictions", rs.get("total_predictions",0))
        c2.metric("Fraud Detected", rs.get("fraud_detected",0))
        c3.metric("Fraud Rate", f"{rs.get('fraud_rate',0):.1f}%")
        c4.metric("AUC Score", perf.get("roc_auc","N/A"))
    except:
        st.warning("Could not load metrics")

    try:
        history = requests.get(f"{API_URL}/history?limit=200", timeout=3).json()
        df = pd.DataFrame(history.get("predictions", []))

        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])
            c1, c2 = st.columns(2)

            with c1:
                counts = df['prediction'].value_counts()
                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    title="Prediction Distribution",
                    color_discrete_map={"FRAUD":"#ff4b4b","LEGITIMATE":"#00c853"}
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.histogram(
                    df, x="amount", color="prediction",
                    title="Amount Distribution",
                    color_discrete_map={"FRAUD":"#ff4b4b","LEGITIMATE":"#00c853"}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Fraud over time
            df_time = df.set_index('created_at').resample('1min')['prediction'].apply(
                lambda x: (x == 'FRAUD').sum()
            ).reset_index()
            fig = px.line(
                df_time, x='created_at', y='prediction',
                title="Fraud Detections Over Time",
                labels={'prediction': 'Fraud Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No predictions yet — make some predictions first!")
    except:
        st.warning("Could not load history")

    if st.button("🔄 Refresh"):
        st.rerun()

# ── Page: History ─────────────────────────────
elif page == "📜 History":
    st.title("📜 Prediction History")
    try:
        history = requests.get(f"{API_URL}/history?limit=100", timeout=3).json()
        df = pd.DataFrame(history.get("predictions", []))
        if not df.empty:
            df['fraud_probability'] = (df['fraud_probability']*100).round(2).astype(str) + "%"
            st.dataframe(
                df[['transaction_id','amount','category','prediction',
                    'fraud_probability','confidence','created_at']],
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"Showing {len(df)} predictions")
        else:
            st.info("No history yet.")
    except:
        st.error("Could not load history")

# ── Page: Model Info ──────────────────────────
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")
    try:
        metrics = requests.get(f"{API_URL}/metrics", timeout=3).json()
        perf = metrics.get("performance", {})
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Model Details")
            st.json({
                "Model": "XGBoost",
                "Technique": "SMOTE",
                "Dataset": "Kaggle CC Transactions",
                "Features": perf.get("features", [])
            })
        with c2:
            st.subheader("Performance")
            for k, v in perf.items():
                if k != "features":
                    st.metric(k.replace("_", " ").title(), v)
    except:
        st.error("Could not load model info")