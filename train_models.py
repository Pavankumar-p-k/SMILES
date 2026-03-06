"""
Train per-endpoint XGBoost ADMET models and save predictor-compatible artifacts.
"""

import argparse
import logging
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "admet_multitask.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "results"

ENDPOINT_CANONICAL = {
    "dili": "DILI",
    "bbb": "BBB",
    "ames": "Ames",
    "herg": "hERG",
}


def canonical_endpoint_name(raw_name: str) -> str:
    return ENDPOINT_CANONICAL.get(raw_name.strip().lower(), raw_name.strip())


def featurize_smiles(smiles_list, n_bits=2048, radius=2):
    """Convert a sequence of SMILES into Morgan-fingerprint feature vectors."""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning("Invalid SMILES: %s", smiles)
            features.append([0] * n_bits)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)
    return np.array(features)


def roc_auc_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    """Compute ROC-AUC with bootstrap confidence intervals."""
    if len(np.unique(y_true)) < 2:
        logging.warning("Single-class target. ROC AUC undefined.")
        return None, None, None
    auc = roc_auc_score(y_true, y_pred)
    bootstrapped_scores = []
    for _ in range(n_bootstrap):
        y_resample, pred_resample = resample(y_true, y_pred)
        if len(np.unique(y_resample)) < 2:
            continue
        bootstrapped_scores.append(roc_auc_score(y_resample, pred_resample))
    if not bootstrapped_scores:
        return auc, None, None
    lower = np.percentile(bootstrapped_scores, ((1 - alpha) / 2) * 100)
    upper = np.percentile(bootstrapped_scores, (alpha + (1 - alpha) / 2) * 100)
    return auc, lower, upper


def train_endpoint(df, endpoint, smiles_col, models_dir, analysis_dir):
    """Train and persist one endpoint model."""
    canonical_name = canonical_endpoint_name(endpoint)
    logging.info("Training endpoint: %s", canonical_name)

    if len(df[endpoint].dropna().unique()) < 2:
        logging.warning(
            "Skipping %s: single-class target %s",
            canonical_name,
            df[endpoint].dropna().unique(),
        )
        return

    endpoint_df = df[[smiles_col, endpoint]].dropna(subset=[endpoint]).copy()
    X = featurize_smiles(endpoint_df[smiles_col].astype(str).tolist())
    y = endpoint_df[endpoint].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc, lower, upper = roc_auc_ci(y_test, y_pred)
    if auc is None:
        logging.info("%s AUC: undefined (single-class test set)", canonical_name)
    elif lower is None or upper is None:
        logging.info("%s AUC: %.3f", canonical_name, auc)
    else:
        logging.info("%s AUC: %.3f (CI %.3f-%.3f)", canonical_name, auc, lower, upper)

    ep_analysis_dir = Path(analysis_dir) / canonical_name
    ep_analysis_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=[f"FP{i}" for i in range(X_train.shape[1])],
        show=False,
    )
    plt.title(f"SHAP Summary - {canonical_name}")
    plt.tight_layout()
    plt.savefig(ep_analysis_dir / f"shap_summary_{canonical_name}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=[f"FP{i}" for i in range(X_train.shape[1])],
        show=False,
    )
    plt.title(f"SHAP Waterfall - {canonical_name} (Sample 0)")
    plt.tight_layout()
    plt.savefig(ep_analysis_dir / f"shap_waterfall_{canonical_name}.png")
    plt.close()

    model_path = Path(models_dir) / f"{canonical_name}_model.pkl"
    joblib.dump(model, model_path)
    logging.info("Model saved: %s", model_path)


def train_all(data_file, models_dir=DEFAULT_MODELS_DIR, analysis_dir=DEFAULT_ANALYSIS_DIR):
    """Train all endpoints found in the provided dataset."""
    df = pd.read_csv(data_file)
    columns_lower = {col.lower(): col for col in df.columns}
    if "smiles" not in columns_lower:
        raise ValueError("Dataset must contain a smiles/SMILES column.")
    smiles_col = columns_lower["smiles"]

    endpoints = [col for col in df.columns if col != smiles_col]
    if not endpoints:
        raise ValueError("No endpoint columns found in dataset.")

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Found endpoints: %s", endpoints)
    for endpoint in endpoints:
        logging.info("=" * 50)
        train_endpoint(
            df=df,
            endpoint=endpoint,
            smiles_col=smiles_col,
            models_dir=models_dir,
            analysis_dir=analysis_dir,
        )


def main():
    parser = argparse.ArgumentParser(description="Train per-endpoint ADMET models.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV with smiles + endpoint columns",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory for predictor-compatible model artifacts",
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Directory for SHAP plots and analysis artifacts",
    )
    args = parser.parse_args()

    train_all(data_file=args.data, models_dir=args.models_dir, analysis_dir=args.analysis_dir)


if __name__ == "__main__":
    main()
