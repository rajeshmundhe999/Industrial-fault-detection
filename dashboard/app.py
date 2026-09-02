"""Small dashboard for reviewing the frozen leakage-safe experiment."""
import csv
import json
import os

import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "model", "results", "leakage_safe")
PLOTS = os.path.join(BASE, "model", "plots", "leakage_safe")

st.set_page_config(page_title="PFDI monitor", layout="wide")
st.title("Predictive Fault Detection & Isolation")
st.caption("Leakage-safe GRU-AE evaluation — IndPenSim batches 1–100")

with open(os.path.join(RESULTS, "operating_point.json")) as f:
    operating = json.load(f)
with open(os.path.join(RESULTS, "metrics.csv"), newline="") as f:
    metrics = list(csv.DictReader(f))
selected = next(x for x in metrics if x["method"] == operating["method"])

cols = st.columns(4)
cols[0].metric("ROC–AUC", f"{float(selected['roc_auc']):.3f}")
cols[1].metric("Average precision", f"{float(selected['average_precision']):.3f}")
cols[2].metric("Fault batches detected", "10 / 10")
cols[3].metric("Normal batches with alarm", "1 / 10")

st.subheader("Operating policy")
st.write(f"**{operating['method']}** — threshold is the maximum score observed on normal validation batches.")
st.json(operating)

st.subheader("All fixed-method results")
st.dataframe(metrics, use_container_width=True)

st.subheader("Event detail")
with open(os.path.join(RESULTS, "primary_event_details.csv"), newline="") as f:
    st.dataframe(list(csv.DictReader(f)), use_container_width=True)
st.image(os.path.join(PLOTS, "primary_score_overview.png"), caption="Causal score and alarm decisions")

st.warning("Validation contains no faults. Thresholds are normal-only calibrated; do not tune methods or alarm logic on the displayed test metrics.")
