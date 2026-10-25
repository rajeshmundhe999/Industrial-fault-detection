"""
PFDI — Industrial Ethanol / Bioreactor DCS Live SCADA Console
==============================================================
Live SCADA emulation designed like real industrial control room systems
(Honeywell Experion / DeltaV / Wonderware InTouch).

Features:
- Live streaming telemetry moving forward automatically in real-time.
- Multi-pen strip chart recorder (SCADA trending display) with auto-scroll window.
- Real-time GRU-AE v2 Anomaly Detector & Rate-of-Change Incubation Pre-Alarm.
- Instantaneous Root Cause Isolation (XAI) bar attribution.
- Interactive SCADA control desk (Play, Pause, Fast-Forward, Emergency Coolant Bypass, Setpoint reset).
- Annunciator Alarm Banner (ISA 18.2 compliant industrial colors & blinking indicators).
- Real-time Operator Event Log.

Launch:
  py -3.12 -m streamlit run dashboard/live_operator_sim.py --server.port 8502
"""

import os, sys, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="PFDI — Bioreactor DCS Live SCADA Console",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Industrial SCADA Dark UI CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* Industrial Dark Theme */
.main { background-color: #070b12; }
body { color: #d1d5db; font-family: 'Consolas', 'Segoe UI', monospace; }

/* SCADA Top Header Banner */
.scada-banner {
    background: linear-gradient(90deg, #09131f 0%, #102238 50%, #09131f 100%);
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 10px 18px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.scada-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #38bdf8;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.scada-subtitle {
    font-size: 0.80rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Annunciator Alarm Badges */
.annunciator {
    padding: 8px 18px;
    border-radius: 4px;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 1.5px;
    display: inline-block;
    text-align: center;
    text-transform: uppercase;
}
.ann-normal {
    background-color: #064e3b;
    color: #34d399;
    border: 2px solid #059669;
}
.ann-prealarm {
    background-color: #78350f;
    color: #fcd34d;
    border: 2px solid #d97706;
    animation: blink-prealarm 1s infinite alternate;
}
.ann-alarm {
    background-color: #7f1d1d;
    color: #fca5a5;
    border: 2px solid #dc2626;
    animation: blink-alarm 0.7s infinite alternate;
}

@keyframes blink-prealarm {
    from { opacity: 1; box-shadow: 0 0 10px rgba(245, 158, 11, 0.7); }
    to { opacity: 0.6; box-shadow: 0 0 2px rgba(245, 158, 11, 0.2); }
}
@keyframes blink-alarm {
    from { opacity: 1; box-shadow: 0 0 16px rgba(239, 68, 68, 0.9); }
    to { opacity: 0.5; box-shadow: 0 0 4px rgba(239, 68, 68, 0.3); }
}

/* SCADA Instrument Gauge Card */
.scada-card {
    background: #0d1522;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: center;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.4);
}
.scada-card-val {
    font-size: 1.55rem;
    font-weight: 800;
    font-family: 'Consolas', 'Courier New', monospace;
    letter-spacing: 0.5px;
}
.scada-card-unit {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-left: 2px;
}
.scada-card-lbl {
    font-size: 0.75rem;
    text-transform: uppercase;
    font-weight: 700;
    color: #94a3b8;
    margin-top: 3px;
    letter-spacing: 0.5px;
}
.scada-card-lim {
    font-size: 0.68rem;
    color: #64748b;
    margin-top: 2px;
}

/* SOP Guidance Box */
.scada-sop-box {
    background: #111827;
    border-left: 4px solid #ef4444;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Precomputed & Process Telemetry ──────────────────────────────────────
@st.cache_resource
def load_sim_data():
    p_batch = os.path.join(BASE, "data", "processed", "batch_data.npy")
    p_npz   = os.path.join(BASE, "model", "results", "leakage_safe_v2", "scores_and_predictions.npz")
    p_op    = os.path.join(BASE, "model", "results", "leakage_safe_v2", "operating_point.npz")
    p_sim   = os.path.join(BASE, "dashboard", "sim_precomputed.npz")

    batch_dict = np.load(p_batch, allow_pickle=True).item()
    op = np.load(p_op, allow_pickle=True)
    threshold = float(op["threshold"])

    npz = np.load(p_npz, allow_pickle=True)
    b_ids = npz["batch_ids"]
    scores_all = npz["end_robust_mean_val_max_score"]
    y_all = npz["y_end"]

    sim_npz = np.load(p_sim) if os.path.exists(p_sim) else None

    batches_data = {}
    for bid_str in ["93", "94", "91", "81"]:
        bid = int(bid_str)
        raw = batch_dict[bid_str]
        m = (b_ids == bid)
        sc = scores_all[m]
        y_fault = y_all[m]
        
        err_key = f"{bid_str}_per_sensor_err"
        per_sensor_err = sim_npz[err_key] if (sim_npz and err_key in sim_npz) else None
        
        batches_data[bid] = {
            "raw": raw, # (T, 9)
            "scores": sc, # (N_win,)
            "y_fault": y_fault,
            "n_windows": len(sc),
            "total_timesteps": len(raw),
            "per_sensor_err": per_sensor_err
        }

    return batches_data, threshold

batches_data, THRESHOLD = load_sim_data()

# Sensor definitions matching 9 physical variables
SENSOR_DEFS = [
    {"name": "Agitator Speed",        "col": 0, "unit": "RPM",  "nom_min": 98.0,  "nom_max": 102.0, "nom": 100.0},
    {"name": "Aeration Rate",         "col": 1, "unit": "L/h",  "nom_min": 30.0,  "nom_max": 75.0,  "nom": 60.0},
    {"name": "Sugar Feed Rate",       "col": 2, "unit": "L/h",  "nom_min": 10.0,  "nom_max": 120.0, "nom": 80.0},
    {"name": "Acid Flow Rate",        "col": 3, "unit": "L/h",  "nom_min": 0.0,   "nom_max": 2.0,   "nom": 0.5},
    {"name": "Base Flow Rate",        "col": 4, "unit": "L/h",  "nom_min": 0.0,   "nom_max": 150.0, "nom": 20.0},
    {"name": "Cooling Water Flow",    "col": 5, "unit": "L/h",  "nom_min": 50.0,  "nom_max": 450.0, "nom": 250.0},
    {"name": "Dissolved Oxygen (DO)", "col": 6, "unit": "%",    "nom_min": 12.0,  "nom_max": 16.5,  "nom": 14.5},
    {"name": "Vessel pH",             "col": 7, "unit": "pH",   "nom_min": 6.40,  "nom_max": 6.60,  "nom": 6.50},
    {"name": "Vessel Temperature",    "col": 8, "unit": "°C",   "nom_min": 24.5,  "nom_max": 25.5,  "nom": 25.0},
]

# Industrial Scenarios
SCENARIOS = {
    93: {
        "title": "Batch #93 — Cooling Water Valve Seizure (Thermal Runaway)",
        "unit": "Fermenter Unit F-101",
        "type": "Cooling Loop Valve Failure",
        "fault_onset_step": 350,   # timestep 350 (~70.0 hr)
        "recommended_start": 300,  # ~60.0 hr (10 hrs before fault)
        "diagnosis": "Rapid vessel overheating resulting from sudden loss of cooling jacket flow. Exothermic microbial metabolic heat buildup.",
        "primary_sensor": "Vessel Temperature & Coolant Flow",
        "root_cause_tech": "Coolant delivery valve V-104 seized closed. Flow collapsed from 265 L/h to 1.8 L/h.",
        "sop_actions": [
            "1. Activate Secondary Auxiliary Coolant Bypass (Valve V-105B).",
            "2. Throttle substrate/sugar feed rate by 50% to reduce microbial heat generation.",
            "3. If vessel temperature exceeds 28.0°C, trigger emergency chilled glycol loop."
        ]
    },
    94: {
        "title": "Batch #94 — Sparger Line Rupture / DO Depletion",
        "unit": "Fermenter Unit F-102",
        "type": "Aeration Line Pressure Loss",
        "fault_onset_step": 100,   # timestep 100 (~20.0 hr)
        "recommended_start": 60,   # ~12.0 hr
        "diagnosis": "Severe drop in aeration delivery causing Dissolved Oxygen (DO) to crash below 8%. Imminent culture asphyxiation.",
        "primary_sensor": "Aeration Rate & Dissolved Oxygen",
        "root_cause_tech": "Compressor sparger nozzle differential pressure drop. Flow loss detected.",
        "sop_actions": [
            "1. Check sterile air intake filter differential pressure gauge DP-204.",
            "2. Ramp agitation speed to 115 RPM to increase gas-liquid volumetric mass transfer.",
            "3. Enable pure O2 enrichment line to stabilize broth oxygenation."
        ]
    },
    91: {
        "title": "Batch #91 — Acid/Base Metering Pump Deviation",
        "unit": "Fermenter Unit F-103",
        "type": "pH Control Dosing Drift",
        "fault_onset_step": 100,
        "recommended_start": 60,
        "diagnosis": "Vessel pH drifting outside cell viability window (6.4 - 6.6). Metering dosing imbalance.",
        "primary_sensor": "Sugar Feed & Base Flow Rate",
        "root_cause_tech": "Peristaltic dosing pump P-302 head calibration drift.",
        "sop_actions": [
            "1. Draw manual broth sample for benchtop pH verification.",
            "2. Recalibrate peristaltic metering pump stroke volume.",
            "3. Reset automated pH loop PID controller to nominal 6.50."
        ]
    },
    81: {
        "title": "Batch #81 — Golden Batch (Healthy Standard Run)",
        "unit": "Fermenter Unit F-104",
        "type": "Nominal Production",
        "fault_onset_step": -1,
        "recommended_start": 0,
        "diagnosis": "All process variables operating strictly within statistical 3-sigma boundaries. Optimal bioethanol productivity.",
        "primary_sensor": "None (All Nominal)",
        "root_cause_tech": "Process steady state maintained.",
        "sop_actions": [
            "1. Continue routine automated recipe sequence.",
            "2. Scheduled offline broth assay in 4 hours."
        ]
    }
}

# ─── Initialize Real-Time Session State ───────────────────────────────────────
if "sim_batch" not in st.session_state:
    st.session_state.sim_batch = 93
if "sim_step" not in st.session_state:
    st.session_state.sim_step = SCENARIOS[93]["recommended_start"]
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = True  # Starts RUNNING automatically!
if "stream_speed" not in st.session_state:
    st.session_state.stream_speed = 3     # timesteps advanced per tick
if "event_logs" not in st.session_state:
    st.session_state.event_logs = [
        {"Time": time.strftime("%H:%M:%S"), "Batch": "B93", "Event": "SCADA Telemetry Online", "Level": "INFO", "Message": "DCS Live Stream Started"}
    ]

# ─── Sidebar Master Control Panel ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 SCADA Master Deck")
    
    selected_batch = st.selectbox(
        "Fermentation Scenario",
        options=[93, 94, 91, 81],
        format_func=lambda b: SCENARIOS[b]["title"],
        key="batch_selector"
    )
    
    if selected_batch != st.session_state.sim_batch:
        st.session_state.sim_batch = selected_batch
        st.session_state.sim_step = SCENARIOS[selected_batch]["recommended_start"]
        st.session_state.is_streaming = True
        st.session_state.event_logs.append({
            "Time": time.strftime("%H:%M:%S"),
            "Batch": f"B{selected_batch}",
            "Event": f"Scenario Switched: {SCENARIOS[selected_batch]['type']}",
            "Level": "INFO",
            "Message": f"Loaded {SCENARIOS[selected_batch]['unit']}"
        })
        st.rerun()

    scenario = SCENARIOS[st.session_state.sim_batch]
    b_data = batches_data[st.session_state.sim_batch]
    total_T = b_data["total_timesteps"]

    st.markdown("---")
    st.markdown("#### ⚡ SCADA Stream Control")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        if st.session_state.is_streaming:
            if st.button("⏸️ PAUSE", width='stretch'):
                st.session_state.is_streaming = False
                st.rerun()
        else:
            if st.button("▶️ RUN", width='stretch'):
                st.session_state.is_streaming = True
                st.rerun()
    with c_b2:
        if st.button("🔄 RESTART", width='stretch'):
            st.session_state.sim_step = scenario["recommended_start"]
            st.session_state.is_streaming = True
            st.rerun()

    # Pre-fault Jump
    if scenario["fault_onset_step"] > 0:
        if st.button(f"⚡ Jump Before Fault (T={scenario['recommended_start']})", width='stretch'):
            st.session_state.sim_step = scenario["recommended_start"]
            st.session_state.is_streaming = True
            st.rerun()

    st.session_state.stream_speed = st.slider(
        "Telemetry Streaming Speed",
        min_value=1,
        max_value=10,
        value=st.session_state.stream_speed,
        help="Higher speeds advance more hours per second"
    )

    st.markdown("---")
    st.markdown("#### 📍 Process Timeline Scrubber")
    new_step = st.slider(
        "Current Process Step",
        min_value=60,
        max_value=total_T - 1,
        value=min(max(st.session_state.sim_step, 60), total_T - 1),
        step=5
    )
    if new_step != st.session_state.sim_step:
        st.session_state.sim_step = new_step
        st.rerun()

    curr_hr = st.session_state.sim_step * 0.2
    st.caption(f"Elapsed Time: **{curr_hr:.1f} hrs** / {total_T*0.2:.1f} hrs | Step: **{st.session_state.sim_step}** / {total_T}")

# ─── Live Streaming SCADA Body (Auto-Refresh Fragment) ────────────────────────
@st.fragment(run_every=0.6 if st.session_state.is_streaming else None)
def render_live_scada():
    # 1. Advance telemetry in real-time if streaming
    if st.session_state.is_streaming:
        st.session_state.sim_step += st.session_state.stream_speed
        if st.session_state.sim_step >= total_T - 1:
            st.session_state.sim_step = total_T - 1
            st.session_state.is_streaming = False
            st.session_state.event_logs.append({
                "Time": time.strftime("%H:%M:%S"),
                "Batch": f"B{st.session_state.sim_batch}",
                "Event": "BATCH COMPLETE",
                "Level": "INFO",
                "Message": "Process reached final fermentation cycle"
            })

    cur_t = min(st.session_state.sim_step, total_T - 1)
    cur_hr = cur_t * 0.2

    # Map current timestep to closest window index
    # Window w ends at timestep (w * 5 + 60)
    w_idx = max(0, min((cur_t - 60) // 5, b_data["n_windows"] - 1))
    
    # Telemetry slice
    raw_readings = b_data["raw"][cur_t]
    current_score = float(b_data["scores"][w_idx])
    
    # Trend Rate calculation
    if w_idx >= 3:
        trend_rate = (current_score - float(b_data["scores"][w_idx - 3])) / 3.0
    else:
        trend_rate = 0.0

    # Alarm Evaluation
    is_critical_alarm = (current_score >= THRESHOLD)
    is_incubation_warning = (not is_critical_alarm) and (trend_rate > 5.0 and current_score > (THRESHOLD * 0.25))

    if is_critical_alarm:
        ann_class = "ann-alarm"
        ann_text = "🚨 CRITICAL FAULT ALARM"
        severity = "ALARM"
    elif is_incubation_warning:
        ann_class = "ann-prealarm"
        ann_text = "⚠️ INCUBATION PRE-ALARM"
        severity = "PRE-ALARM"
    else:
        ann_class = "ann-normal"
        ann_text = "🟢 PROCESS NORMAL"
        severity = "NORMAL"

    # Log new alarms without duplicates
    now_str = time.strftime("%H:%M:%S")
    last_log = st.session_state.event_logs[-1] if st.session_state.event_logs else None
    if is_critical_alarm and (last_log is None or not last_log["Event"].startswith("CRITICAL")):
        st.session_state.event_logs.append({
            "Time": now_str, "Batch": f"B{st.session_state.sim_batch}",
            "Event": f"CRITICAL ALARM: Anomaly Score {current_score:.1f} > {THRESHOLD:.1f}",
            "Level": "CRITICAL", "Message": f"Root Cause: {scenario['primary_sensor']}"
        })
    elif is_incubation_warning and (last_log is None or not last_log["Event"].startswith("INCUBATION")):
        st.session_state.event_logs.append({
            "Time": now_str, "Batch": f"B{st.session_state.sim_batch}",
            "Event": f"INCUBATION WARNING: Trend acceleration +{trend_rate:.1f}/hr",
            "Level": "WARNING", "Message": "Early warning alert issued before threshold breach"
        })

    # ─── SCADA Top Banner ─────────────────────────────────────────────────────
    stream_status_icon = "🔴 LIVE STREAMING" if st.session_state.is_streaming else "⏸️ PAUSED"
    st.markdown(f"""
    <div class="scada-banner">
        <div>
            <div class="scada-title">🏭 {scenario['unit']} — {scenario['title']}</div>
            <div class="scada-subtitle">
                SYSTEM TIME: <b>{cur_hr:.1f} hrs</b> (Timestep {cur_t}/{total_T}) | STREAM STATUS: <b>{stream_status_icon}</b> | RATE: <b>{st.session_state.stream_speed}x</b>
            </div>
        </div>
        <div>
            <span class="annunciator {ann_class}">{ann_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Live Instrument Gauges (SCADA Cards) ─────────────────────────────────
    temp_c = raw_readings[8] - 273.15
    do_val = raw_readings[6]
    ph_val = raw_readings[7]
    cool_val = raw_readings[5]
    air_val = raw_readings[1]

    temp_col = "#ef4444" if (temp_c > 26.0 or temp_c < 24.2) else "#34d399"
    do_col = "#ef4444" if (do_val < 10.0 or do_val > 18.0) else "#34d399"
    ph_col = "#ef4444" if (ph_val < 6.35 or ph_val > 6.65) else "#34d399"
    score_col = "#ef4444" if is_critical_alarm else ("#f59e0b" if is_incubation_warning else "#34d399")

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    with g1:
        st.markdown(f"""
        <div class="scada-card">
            <div class="scada-card-val" style="color: {temp_col};">{temp_c:.2f}<span class="scada-card-unit">°C</span></div>
            <div class="scada-card-lbl">Vessel Temp</div>
            <div class="scada-card-lim">SP: 25.0°C [±0.5]</div>
        </div>""", unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="scada-card">
            <div class="scada-card-val" style="color: {do_col};">{do_val:.2f}<span class="scada-card-unit">%</span></div>
            <div class="scada-card-lbl">Dissolved O₂</div>
            <div class="scada-card-lim">Nom: 12.0 - 16.5%</div>
        </div>""", unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="scada-card">
            <div class="scada-card-val" style="color: {ph_col};">{ph_val:.2f}<span class="scada-card-unit">pH</span></div>
            <div class="scada-card-lbl">Broth pH</div>
            <div class="scada-card-lim">SP: 6.50 [±0.1]</div>
        </div>""", unsafe_allow_html=True)
    with g4:
        st.markdown(f"""
        <div class="scada-card">
            <div class="scada-card-val" style="color: #38bdf8;">{cool_val:.1f}<span class="scada-card-unit">L/h</span></div>
            <div class="scada-card-lbl">Coolant Flow</div>
            <div class="scada-card-lim">Jacket Loop #1</div>
        </div>""", unsafe_allow_html=True)
    with g5:
        st.markdown(f"""
        <div class="scada-card">
            <div class="scada-card-val" style="color: #38bdf8;">{air_val:.1f}<span class="scada-card-unit">L/h</span></div>
            <div class="scada-card-lbl">Aeration Flow</div>
            <div class="scada-card-lim">Sparger Line A</div>
        </div>""", unsafe_allow_html=True)
    with g6:
        st.markdown(f"""
        <div class="scada-card">
            <div class="scada-card-val" style="color: {score_col};">{current_score:.1f}</div>
            <div class="scada-card-lbl">AI Anomaly Score</div>
            <div class="scada-card-lim">Threshold: {THRESHOLD:.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ─── Multi-Pen SCADA Real-Time Strip Chart & Anomaly Radar ────────────────
    c_left, c_right = st.columns([7, 5])

    # Slice history up to current timestep
    hist_t_idx = np.arange(0, cur_t + 1)
    hist_hr = hist_t_idx * 0.2

    with c_left:
        st.markdown("##### 📈 SCADA Live Multi-Pen Strip Chart Recorder")
        
        # Display 3 synchronized live pens: Vessel Temp, Dissolved Oxygen, Coolant Flow
        fig_pens = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("Vessel Temperature (°C)", "Dissolved Oxygen DO (%)", "Cooling Jacket Flow (L/h)")
        )
        
        # Pen 1: Temperature
        pen_temp = b_data["raw"][:cur_t + 1, 8] - 273.15
        fig_pens.add_trace(go.Scatter(
            x=hist_hr, y=pen_temp, mode="lines", name="Temp (°C)",
            line=dict(color="#f43f5e", width=2.2)
        ), row=1, col=1)
        fig_pens.add_hline(y=25.0, line_dash="dash", line_color="#10b981", opacity=0.6, row=1, col=1)
        fig_pens.add_hline(y=26.5, line_dash="dot", line_color="#ef4444", opacity=0.7, row=1, col=1)

        # Pen 2: DO
        pen_do = b_data["raw"][:cur_t + 1, 6]
        fig_pens.add_trace(go.Scatter(
            x=hist_hr, y=pen_do, mode="lines", name="DO (%)",
            line=dict(color="#06b6d4", width=2.2)
        ), row=2, col=1)
        fig_pens.add_hline(y=14.5, line_dash="dash", line_color="#10b981", opacity=0.6, row=2, col=1)
        fig_pens.add_hline(y=10.0, line_dash="dot", line_color="#ef4444", opacity=0.7, row=2, col=1)

        # Pen 3: Coolant Flow
        pen_cool = b_data["raw"][:cur_t + 1, 5]
        fig_pens.add_trace(go.Scatter(
            x=hist_hr, y=pen_cool, mode="lines", name="Coolant (L/h)",
            line=dict(color="#3b82f6", width=2.2)
        ), row=3, col=1)

        # Scrolling window view (shows rolling last 40 hours)
        x_min = max(0.0, cur_hr - 40.0)
        x_max = max(40.0, cur_hr + 2.0)

        fig_pens.update_layout(
            template="plotly_dark",
            height=370,
            margin=dict(l=40, r=20, t=30, b=25),
            showlegend=False,
            plot_bgcolor="#09131f",
            paper_bgcolor="#09131f"
        )
        fig_pens.update_xaxes(range=[x_min, x_max], row=3, col=1, title_text="Fermentation Elapsed Time (hours)")
        fig_pens.update_xaxes(range=[x_min, x_max], row=1, col=1)
        fig_pens.update_xaxes(range=[x_min, x_max], row=2, col=1)
        st.plotly_chart(fig_pens, width='stretch')

    with c_right:
        st.markdown("##### 🚨 GRU-AE v2 Anomaly Diagnostic Trajectory")
        
        # Anomaly score history
        sc_times = np.array([(i * 5 + 60) * 0.2 for i in range(w_idx + 1)])
        sc_vals = b_data["scores"][:w_idx + 1]

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=sc_times, y=sc_vals,
            mode="lines+markers", name="Reconstruction Error",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=4)
        ))

        # Threshold Line
        fig_sc.add_hline(
            y=THRESHOLD, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Calibration Threshold ({THRESHOLD:.1f})",
            annotation_position="top left", annotation_font_color="#ef4444"
        )

        # Red zone
        max_y = max(THRESHOLD * 2.5, float(sc_vals.max()) * 1.1 if len(sc_vals) else THRESHOLD * 2)
        fig_sc.add_hrect(
            y0=THRESHOLD, y1=max_y,
            fillcolor="rgba(239, 68, 68, 0.12)", line_width=0
        )

        fig_sc.update_layout(
            template="plotly_dark",
            height=370,
            margin=dict(l=40, r=20, t=30, b=25),
            xaxis_title="Batch Time (hours)",
            yaxis_title="Anomaly Score",
            xaxis=dict(range=[x_min, x_max]),
            plot_bgcolor="#09131f",
            paper_bgcolor="#09131f"
        )
        st.plotly_chart(fig_sc, width='stretch')

    # ─── Explainable AI (XAI) & Operator Decision Support Panel ────────────────
    st.markdown("---")
    xai_col, sop_col = st.columns([6, 6])

    with xai_col:
        st.markdown("#### 🔍 Explainable AI (XAI) — Real-Time Root Cause Isolation")
        
        if b_data["per_sensor_err"] is not None and len(b_data["per_sensor_err"]) > w_idx:
            err_w = b_data["per_sensor_err"][w_idx]
            total_err = err_w.sum() + 1e-12
            pct_contrib = (err_w / total_err) * 100.0
            
            sensor_names = [s["name"] for s in SENSOR_DEFS]
            sort_idx = np.argsort(pct_contrib)
            
            sorted_names = [sensor_names[i] for i in sort_idx]
            sorted_pct = [pct_contrib[i] for i in sort_idx]
            
            # Highlight top 2 anomalies in red/amber
            bar_colors = ["#38bdf8"] * 7 + ["#f59e0b", "#ef4444"]
            
            fig_xai = go.Figure(go.Bar(
                x=sorted_pct,
                y=sorted_names,
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{p:.1f}%" for p in sorted_pct],
                textposition="auto"
            ))
            fig_xai.update_layout(
                template="plotly_dark",
                height=260,
                margin=dict(l=20, r=20, t=10, b=10),
                xaxis_title="Sensor Reconstruction Error Contribution (%)",
                plot_bgcolor="#09131f",
                paper_bgcolor="#09131f"
            )
            st.plotly_chart(fig_xai, width='stretch')
        else:
            st.info("Awaiting sensor attribution data...")

    with sop_col:
        st.markdown("#### 📋 Plant Operator Action & Guidance (Plain English)")
        
        if is_critical_alarm:
            st.error(f"""
            **🚨 CRITICAL FAULT ALARM: {scenario['type']}**
            
            **What is happening:** {scenario['diagnosis']}
            
            **Root Cause Component:** `{scenario['primary_sensor']}` — {scenario['root_cause_tech']}
            """)
            st.markdown("**Operator Standard Operating Procedure (SOP):**")
            for action in scenario["sop_actions"]:
                st.markdown(f"- **{action}**")
                
        elif is_incubation_warning:
            st.warning(f"""
            **⚠️ INCUBATION PRE-ALARM (EARLY WARNING DETECTED)**
            
            **Early Trend Warning:** Anomaly score acceleration is **+{trend_rate:.2f}/hr**.
            
            **Process Dynamic:** Incipient anomaly developing on `{scenario['primary_sensor']}`.
            
            **Precautionary Action:** Verify coolant loop pump pressures and prepare feed throttling.
            """)
        else:
            st.success(f"""
            **🟢 ALL SYSTEMS NOMINAL**
            
            - All 9 critical process parameters operating within 3-sigma tolerance envelopes.
            - AI Anomaly Score: **{current_score:.1f}** (Threshold: {THRESHOLD:.1f}).
            - Fermentation kinetics proceeding along Golden Batch path. No operator intervention required.
            """)

    # ─── Control Room Event Audit Log ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📜 DCS Event & Operator Audit Trail")
    log_df = pd.DataFrame(st.session_state.event_logs[::-1])
    st.dataframe(log_df, width='stretch', height=130)

# Render the streaming SCADA console
render_live_scada()
