import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
DESCRIPTOR_COUNT = 8
FEATURE_DIM = FINGERPRINT_BITS + DESCRIPTOR_COUNT

PROJECT_ROOT = Path(__file__).parent
DATA_LARGE_DIR = PROJECT_ROOT / "data_large"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_rdkit_features(smiles: str) -> Optional[np.ndarray]:
    """Return Morgan bits + 8 RDKit descriptors, or None for invalid SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=FINGERPRINT_RADIUS,
            nBits=FINGERPRINT_BITS,
        )
        fp_array = np.array(fp)
        descriptors = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumAromaticRings(mol) / max(1, Descriptors.RingCount(mol)),
            ],
            dtype=float,
        )
        return np.concatenate([fp_array, descriptors])
    except Exception:
        return None


def featurize_smiles_series(smiles_series: pd.Series) -> np.ndarray:
    """Featurize a smiles series, replacing invalid rows with zeros."""
    features = []
    for idx, smiles in enumerate(smiles_series.astype(str)):
        if idx % 500 == 0:
            print(f"  {idx}/{len(smiles_series)}")
        vector = extract_rdkit_features(smiles)
        if vector is None:
            vector = np.zeros(FEATURE_DIM, dtype=float)
        features.append(vector)
    return np.array(features, dtype=float)


def load_optional_preprocessing(
    models_dir: Path,
) -> Tuple[Optional[object], Optional[object]]:
    """Load optional preprocessing artifacts if both exist."""
    var_threshold_path = models_dir / "var_threshold.pkl"
    scaler_path = models_dir / "scaler.pkl"

    if var_threshold_path.exists() and scaler_path.exists():
        print("Loading preprocessing artifacts...")
        return joblib.load(var_threshold_path), joblib.load(scaler_path)

    print("Preprocessing artifacts not found; continuing without transform.")
    return None, None


def apply_optional_preprocessing(
    X: np.ndarray,
    var_threshold: Optional[object],
    scaler: Optional[object],
) -> np.ndarray:
    """Apply optional variance-threshold + descriptor scaling transforms."""
    transformed = X
    if var_threshold is not None:
        transformed = var_threshold.transform(transformed)
    if scaler is not None and transformed.shape[1] >= DESCRIPTOR_COUNT:
        transformed = transformed.copy()
        transformed[:, -DESCRIPTOR_COUNT:] = scaler.transform(
            transformed[:, -DESCRIPTOR_COUNT:]
        )
    return transformed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_LARGE_DIR / "tox21_train.csv"
    test_path = DATA_LARGE_DIR / "tox21_test.csv"
    if not test_path.exists():
        test_path = DATA_LARGE_DIR / "tox21_valid.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(
            "Neither tox21_test.csv nor tox21_valid.csv exists in data_large/"
        )

    print("Extracting training features...")
    df_train = pd.read_csv(train_path)
    X_train = featurize_smiles_series(df_train["smiles"])

    var_threshold, scaler = load_optional_preprocessing(MODELS_DIR)
    X_train_processed = apply_optional_preprocessing(X_train, var_threshold, scaler)

    print("\nTraining Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_processed)

    print("Extracting test features...")
    df_test = pd.read_csv(test_path)
    X_test = featurize_smiles_series(df_test["smiles"])
    X_test_processed = apply_optional_preprocessing(X_test, var_threshold, scaler)

    anomaly_scores = iso_forest.decision_function(X_test_processed)
    predictions = iso_forest.predict(X_test_processed)

    ad_status = ["Inside AD" if pred == 1 else "Outside AD" for pred in predictions]
    pct_outside = (predictions == -1).sum() / len(predictions) * 100

    print(f"\n{'=' * 60}")
    print("Applicability Domain Analysis")
    print(f"{'=' * 60}")
    print(f"Test compounds inside AD: {(predictions == 1).sum()} ({100 - pct_outside:.1f}%)")
    print(f"Test compounds outside AD: {(predictions == -1).sum()} ({pct_outside:.1f}%)")

    pd.DataFrame(
        {
            "SMILES": df_test["smiles"].values,
            "AD_Status": ad_status,
            "Anomaly_Score": anomaly_scores,
        }
    ).to_csv(RESULTS_DIR / "applicability_domain.csv", index=False)
    joblib.dump(iso_forest, MODELS_DIR / "applicability_domain.pkl")

    print("\nApplicability domain artifacts saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
