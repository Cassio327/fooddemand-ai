from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REPORT_PATH = Path(os.getenv("REPORT_PATH", "reports/latest_metrics.json"))

st.set_page_config(page_title="FoodDemand AI", page_icon="📦", layout="wide")
st.title("FoodDemand AI")
st.caption("Demand forecasting demo backed by the FastAPI inference service.")

if REPORT_PATH.exists():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("data_source") != "kaggle-store-sales":
        st.warning("Synthetic/unverified metrics: integration demo only, not Kaggle results.")
    aggregate = report.get("backtest", {}).get("aggregate", {})
    model = aggregate.get("model", {})
    baseline = aggregate.get("seasonal_naive", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Model RMSLE", f"{model.get('rmsle', 0):.4f}")
    col2.metric("Seasonal baseline RMSLE", f"{baseline.get('rmsle', 0):.4f}")
    improvement = report.get("backtest", {}).get("improvement_vs_seasonal_naive", {}).get("rmsle")
    col3.metric("RMSLE improvement", "not run" if improvement is None else f"{improvement:.1%}")
else:
    st.info("Run the training pipeline to populate real evaluation metrics.")

st.subheader("Try a forecast")
st.caption("This demo requires historical lag inputs; it is not a recursive multi-day forecast.")
with st.form("forecast"):
    col1, col2, col3 = st.columns(3)
    forecast_date = col1.date_input("Date", value=date(2017, 8, 16))
    store_nbr = col2.number_input("Store", min_value=1, value=1)
    family = col3.text_input("Product family", value="GROCERY I")
    onpromotion = st.number_input("Products on promotion", min_value=0, value=0)
    lag_7 = st.number_input("Sales 7 days ago", min_value=0.0, value=100.0)
    lag_14 = st.number_input("Sales 14 days ago", min_value=0.0, value=100.0)
    lag_28 = st.number_input("Sales 28 days ago", min_value=0.0, value=100.0)
    rolling_7 = st.number_input("7-day historical mean", min_value=0.0, value=100.0)
    rolling_28 = st.number_input("28-day historical mean", min_value=0.0, value=100.0)
    submitted = st.form_submit_button("Forecast")

if submitted:
    payload = {
        "date": forecast_date.isoformat(),
        "store_nbr": int(store_nbr),
        "family": family,
        "onpromotion": int(onpromotion),
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "rolling_mean_7": rolling_7,
        "rolling_mean_28": rolling_28,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        output = response.json()
        st.success(f"Predicted sales: {output['predicted_sales']:.2f}")
        st.caption(f"Model version: {output['model_version']}")
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
