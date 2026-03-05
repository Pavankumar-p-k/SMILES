"""Central configuration for ADMET platform."""

from pathlib import Path
from typing import Dict

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ============================================================================
# ML HYPERPARAMETERS
# ============================================================================
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2

# Feature generation
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "verbosity": 0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RANDOMFOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOGISTICREGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ============================================================================
# APPLICABILITY DOMAIN (updated thresholds)
# ============================================================================
AD_THRESHOLDS = {
    "In-Domain": 0.7,
    "Near-Boundary": 0.5,
    "Out-of-Domain": 0.0,
}

# ============================================================================
# RISK AGGREGATION
# ============================================================================
ENDPOINT_WEIGHTS = {
    "DILI": 0.3,
    "hERG": 0.3,
    "Ames": 0.2,
    "BBB": 0.2,
}

RISK_CATEGORIES = {
    "Low": (0.0, 0.33),
    "Moderate": (0.33, 0.66),
    "High": (0.66, 1.0),
}

# ============================================================================
# ENDPOINTS
# ============================================================================
ENDPOINTS = ["DILI", "BBB", "Ames", "hERG"]
XGBOOST_ENDPOINTS = ["DILI", "BBB"]
RANDOMFOREST_ENDPOINTS = ["Ames", "hERG"]

ENDPOINT_INFO = {
    "DILI": {
        "full_name": "Drug-Induced Liver Injury",
        "type": "XGBoost",
        "description": "Hepatotoxicity risk",
    },
    "BBB": {
        "full_name": "Blood-Brain Barrier",
        "type": "XGBoost",
        "description": "CNS penetration risk",
    },
    "Ames": {
        "full_name": "Ames Mutagenicity",
        "type": "RandomForest",
        "description": "Genetic toxicity risk",
    },
    "hERG": {
        "full_name": "hERG Ion Channel",
        "type": "RandomForest",
        "description": "Cardiac toxicity risk",
    },
}

# ============================================================================
# PROBABILITY CALIBRATION
# ============================================================================
CALIBRATION_METHOD = "sigmoid"

# ============================================================================
# DESCRIPTOR NAMES
# ============================================================================
DESCRIPTOR_NAMES = [
    "Molecular Weight",
    "LogP",
    "H-Bond Donors",
    "H-Bond Acceptors",
    "Rotatable Bonds",
    "TPSA",
    "Aromatic Rings",
] + [f"Morgan_FP_{i}" for i in range(FINGERPRINT_BITS)]

# ============================================================================
# LOGGING
# ============================================================================
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# ============================================================================
# UI
# ============================================================================
RISK_COLORS = {
    "Low": "#00B050",  # Green
    "Moderate": "#FFC000",  # Amber
    "High": "#FF0000",  # Red
}

# ============================================================================
# VALIDATION
# ============================================================================
MIN_MOLECULES_TRAIN = 50
MAX_SMILES_LENGTH = 200
