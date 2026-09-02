"""
PFDI PROJECT — MODULE INTERFACES
==================================
These dataclasses define EXACTLY what each module outputs.
Tanish (model), Rajesh (data), and Anand (dashboard) all
import from here. Nobody changes this without team agreement.

Think of this as the API contract between modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class SensorReading:
    """
    One timestep of sensor data from a fermentation vessel.
    Output of: data/loader.py
    Consumed by: model/tranad_fermentation.py
    """
    batch_id: int
    timestep: float                      # hours since batch start (e.g. 12.4)
    timestamp: Optional[datetime]        # real timestamp if available
    sensors: Dict[str, float]           # {sensor_name: value}
    # e.g. {"pH": 6.5, "Temperature": 25.3, "Dissolved_oxygen": 35.2, ...}
    is_faulted_timestep: Optional[int] = None  # 1=fault, 0=normal, None=unknown


@dataclass
class WindowBatch:
    """
    A sliding window of sensor readings ready for model input.
    Output of: data/preprocess.py
    Consumed by: model/tranad_fermentation.py
    Shape: (WINDOW_SIZE, N_SENSORS)
    """
    batch_id: int
    window_start_h: float               # start hour of this window
    window_end_h: float                 # end hour of this window
    data: object                        # numpy array shape (WINDOW_SIZE, N_SENSORS)
    sensor_names: List[str]             # ordered list matching data columns
    label: int                          # 0=normal, 1=fault (for evaluation only)
    fault_type: Optional[str] = None    # which fault type if known


@dataclass
class AnomalyScore:
    """
    Model output for a single window.
    Output of: model/tranad_fermentation.py
    Consumed by: model/isolate.py, dashboard/app.py
    """
    batch_id: int
    window_end_h: float                 # hour at which this score was computed
    score: float                        # reconstruction error (higher = more anomalous)
    is_anomaly: bool                    # True if score > calibrated threshold
    threshold: float                    # the threshold used
    per_sensor_error: Dict[str, float]  # {sensor_name: reconstruction_error}
    confidence: float                   # model confidence 0.0-1.0


@dataclass
class IsolationReport:
    """
    Root-cause attribution for a detected anomaly.
    Output of: model/isolate.py
    Consumed by: dashboard/app.py, paper results
    """
    batch_id: int
    window_end_h: float
    causal_sensor: str                  # sensor with highest SHAP attribution
    shap_values: Dict[str, float]       # {sensor_name: shap_value} for all sensors
    confidence: float                   # how certain we are (0.0-1.0)
    fault_type_prediction: str          # mapped to FAULT_SIGNATURES key
    recommended_action: str             # human-readable intervention text
    hours_before_yield_impact: Optional[float] = None  # early warning window


@dataclass
class BatchReport:
    """
    Complete report for one finished batch.
    Aggregates all AnomalyScores and IsolationReports for one batch.
    Used by: dashboard batch history, paper results table
    """
    batch_id: int
    batch_start_h: float
    batch_end_h: float
    n_timesteps: int
    true_label: int                     # 0=healthy, 1=faulty (from IndPenSim)
    predicted_label: int                # 0=healthy, 1=faulty (our model)
    first_anomaly_detected_h: Optional[float]    # hour of first alert
    fault_type_detected: Optional[str]
    peak_anomaly_score: float
    final_yield_g_L: Optional[float]    # penicillin yield at end of batch
    anomaly_scores: List[AnomalyScore] = field(default_factory=list)
    isolation_reports: List[IsolationReport] = field(default_factory=list)


# ─── RECOMMENDED ACTIONS MAP ─────────────────────────────────────────────────
# Maps fault_type_prediction → operator instruction
# Used by IsolationReport.recommended_action
RECOMMENDED_ACTIONS = {
    "substrate_depletion": (
        "IMMEDIATE: Increase substrate feed rate. Check feed pump operation. "
        "Verify substrate tank level. Monitor CO2 evolution — should rise within 30 min."
    ),
    "oxygen_limitation": (
        "IMMEDIATE: Increase aeration rate and agitation speed. "
        "Check sparger for blockage. Verify dissolved oxygen probe calibration. "
        "Consider reducing feed rate temporarily to lower oxygen demand."
    ),
    "temperature_deviation": (
        "CHECK: Inspect cooling water supply and jacket temperature controller. "
        "Verify temperature probe placement. If temperature rising: "
        "open cooling water valve manually. If falling: check heating system."
    ),
    "ph_deviation": (
        "CHECK: Verify acid/base dosing pump operation. "
        "Inspect pH probe for fouling or drift — recalibrate if needed. "
        "If pH falling: increase base addition. If rising: check CO2 supply."
    ),
    "agitation_failure": (
        "CRITICAL: Check agitator motor and impeller shaft. "
        "Verify electrical supply to agitator. "
        "Increase aeration temporarily to compensate for reduced mixing. "
        "Call maintenance if agitation does not recover within 15 minutes."
    ),
    "unknown": (
        "INVESTIGATE: Multiple parameters deviating simultaneously. "
        "Review all sensor readings manually. Check process historian for recent changes. "
        "Consider conservative intervention: reduce feed rate, maintain DO above 20%."
    ),
}
