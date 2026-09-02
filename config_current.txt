"""
PFDI PROJECT — MASTER CONFIG
==============================
Project: Predictive Fault Detection and Isolation in Industrial
         Batch Processes using Sequential Sensor Data
Base Paper: Petrovski et al. 2023 (GRU-AE on penicillin fermentation)
Architecture: TranAD (Tuli et al., VLDB 2022)
Dataset: IndPenSim (Goldrick et al., 2019) — 100 batch penicillin fermentation

DO NOT HARDCODE THESE VALUES ANYWHERE ELSE IN THE PROJECT.
Every module imports from here.
"""

# ─── DATASET ────────────────────────────────────────────────────────────────

# IndPenSim full variable list (from official help file + Goldrick 2019)
# ONLINE measurements (available in real-time, suitable for monitoring)
ONLINE_SENSORS = [
    "Substrate_feed_rate",      # Fs  — substrate feed rate (L/h)
    "Glucose_concentration",    # Fg  — glucose feed concentration (g/L)
    "Aeration_rate",            # Fa  — aeration rate (L/h)
    "Agitation_rate",           # RPM — agitation speed (rpm)
    "Temperature",              # T   — vessel temperature (°C)
    "pH",                       # pH  — hydrogen ion concentration
    "Dissolved_oxygen",         # DO2 — dissolved oxygen (%)
    "Vessel_volume",            # V   — vessel volume (L)
    "CO2_percent",              # CO2 — CO2 percent in off-gas (%)
    "O2_percent",               # O2  — O2 percent in off-gas (%)
    "Penicillin_yield",         # P   — penicillin concentration (g/L) [TARGET]
    "CER",                      # CER — carbon evolution rate (mmol/L/h)
    "OUR",                      # OUR — oxygen uptake rate (mmol/L/h)
]

# OFFLINE measurements (lab measurements, taken periodically — NOT for real-time)
OFFLINE_SENSORS = [
    "Substrate_concentration",  # S   — substrate conc (g/L), measured offline
    "Biomass_concentration",    # X   — biomass conc (g/L), measured offline
    "PAA_concentration",        # PAA — phenylacetic acid (g/L), measured offline
    "NH3_concentration",        # NH3 — nitrogen conc (g/L), measured offline
    "Viscosity",                # Visc — viscosity (cP), measured offline
]

# SENSORS WE USE FOR FAULT DETECTION (real-time only, no offline)
# These 8 are available continuously — exactly what you'd have in a real distillery
MONITORING_SENSORS = [
    "Substrate_feed_rate",
    "Aeration_rate",
    "Agitation_rate",
    "Temperature",
    "pH",
    "Dissolved_oxygen",
    "CO2_percent",
    "O2_percent",
    "CER",                      # calculated from CO2/O2 — available online
]
N_SENSORS = len(MONITORING_SENSORS)  # 9 sensors

# TARGET variable (what we ultimately want to protect)
TARGET_VARIABLE = "Penicillin_yield"

# ─── BATCH STRUCTURE ────────────────────────────────────────────────────────

# IndPenSim batch structure (Goldrick 2019, confirmed by Frontiers 2024 paper)
TOTAL_BATCHES = 100
HEALTHY_BATCHES = list(
    set(range(1, 101))
    - {3,4,6,9,10,18,19,24,25,28,30,33,34,36,37,40,42,43,44,45,
       47,49,51,54,59,60,91,92,94,95,97,99,100}
)  # 67 healthy batches
UNHEALTHY_BATCHES = [
    3,4,6,9,10,18,19,24,25,28,30,33,34,36,37,40,42,43,44,45,
    47,49,51,54,59,60,91,92,94,95,97,99,100
]  # 33 unhealthy batches (penicillin yield < 20 g/L)

# Explicitly faulted batches (injected faults, not just low yield)
FAULTED_BATCHES = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

# Control strategy groupings
RECIPE_BATCHES = list(range(1, 31))      # 1-30: recipe driven
OPERATOR_BATCHES = list(range(31, 61))   # 31-60: operator controlled
APC_BATCHES = list(range(61, 91))        # 61-90: APC with Raman
FAULT_BATCHES = list(range(91, 101))     # 91-100: injected faults

# Batch timing (IndPenSim standard)
SAMPLING_INTERVAL_H = 0.2       # 12 minutes (0.2 hours) per timestep
AVG_BATCH_DURATION_H = 230.0    # hours — IndPenSim standard batch
MAX_TIMESTEPS_PER_BATCH = 1150  # 230h / 0.2h

# PENICILLIN YIELD THRESHOLD for healthy vs unhealthy classification
YIELD_THRESHOLD_G_L = 20.0      # g/L — below this = unhealthy batch

# ─── TRAIN / TEST SPLIT ─────────────────────────────────────────────────────

# Use first 80 batches for training, last 20 for testing
# Important: train ONLY on batches 1-60 (recipe + operator) to avoid
# data leakage from APC-controlled batches that are more "perfect"
TRAIN_BATCH_IDS = RECIPE_BATCHES + OPERATOR_BATCHES[:20]  # 50 batches
VAL_BATCH_IDS = OPERATOR_BATCHES[20:]                      # 10 batches
TEST_BATCH_IDS = APC_BATCHES[:20] + FAULT_BATCHES          # 30 batches (20 normal + 10 faulted)

# ─── MODEL PARAMETERS ───────────────────────────────────────────────────────

# Sliding window parameters (TranAD uses windows of time)
WINDOW_SIZE = 60        # 60 timesteps = 12 hours of data (60 × 0.2h)
STEP_SIZE = 1           # slide 1 step at a time (aggressive — catches early faults)
PREDICTION_HORIZON = 25 # predict fault 25 steps (5 hours) ahead

# TranAD architecture parameters (from Tuli et al. 2022, adapted for 9 sensors)
D_MODEL = 64            # transformer model dimension
N_HEADS = 4             # attention heads
N_ENCODER_LAYERS = 2    # encoder depth
DROPOUT = 0.1           # dropout rate
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_SEED = 42

# Anomaly thresholding
ANOMALY_PERCENTILE = 85  # POT will refine this, but start here
# POT = Peak Over Threshold (extreme value theory, used by TranAD paper)

# ─── FAULT TYPES ─────────────────────────────────────────────────────────────
# IndPenSim batches 91-100 contain process deviations
# Based on Goldrick 2019 and Petrovski 2023 analysis:
# Faults manifest in these specific sensor signatures:
FAULT_SIGNATURES = {
    "substrate_depletion": {
        "primary_sensor": "Substrate_feed_rate",
        "description": "Substrate feed drops — penicillin production starves",
        "affected_sensors": ["Substrate_feed_rate", "CO2_percent", "OUR", "CER"],
    },
    "oxygen_limitation": {
        "primary_sensor": "Dissolved_oxygen",
        "description": "DO crashes — anaerobic conditions, yeast/mold stress",
        "affected_sensors": ["Dissolved_oxygen", "Aeration_rate", "O2_percent", "OUR"],
    },
    "temperature_deviation": {
        "primary_sensor": "Temperature",
        "description": "Temperature spike or drop — enzyme denaturation or growth inhibition",
        "affected_sensors": ["Temperature", "Dissolved_oxygen", "pH"],
    },
    "ph_deviation": {
        "primary_sensor": "pH",
        "description": "pH drift — inhibits enzyme activity, cell viability drops",
        "affected_sensors": ["pH", "CO2_percent", "CER"],
    },
    "agitation_failure": {
        "primary_sensor": "Agitation_rate",
        "description": "Agitation drops — mixing failure, oxygen transfer limited",
        "affected_sensors": ["Agitation_rate", "Dissolved_oxygen", "CER", "OUR"],
    },
}

# ─── PATHS ──────────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")        # IndPenSim CSVs go here
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # .npy arrays go here
DATA_PLOTS_DIR     = os.path.join(BASE_DIR, "data", "plots")      # EDA plots
MODEL_SAVED_DIR    = os.path.join(BASE_DIR, "model", "saved")     # checkpoints
MODEL_RESULTS_DIR  = os.path.join(BASE_DIR, "model", "results")   # metrics JSON
MODEL_PLOTS_DIR    = os.path.join(BASE_DIR, "model", "plots")     # training curves
PAPER_DIR          = os.path.join(BASE_DIR, "paper")              # LaTeX files

# Create all directories on import
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_PLOTS_DIR,
             MODEL_SAVED_DIR, MODEL_RESULTS_DIR, MODEL_PLOTS_DIR, PAPER_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─── DATA FILE NAMING ────────────────────────────────────────────────────────
# IndPenSim CSV files are named: Batch_1.csv, Batch_2.csv ... Batch_100.csv
def batch_csv_path(batch_id: int) -> str:
    return os.path.join(DATA_RAW_DIR, f"Batch_{batch_id}.csv")

def batch_label(batch_id: int) -> int:
    """Returns 1 if batch is unhealthy (fault), 0 if healthy."""
    return 1 if batch_id in UNHEALTHY_BATCHES else 0
