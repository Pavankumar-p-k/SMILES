"""Train calibrated multi-endpoint ADMET models with scaffold cross-validation.

The training target file must contain:
- `smiles`
- one binary column per endpoint (default: DILI, BBB, Ames, hERG)

Examples
--------
>>> len(scaffold_kfold_indices(["CCO", "CCN", "c1ccccc1"], n_splits=2, random_state=0))
2
>>> sorted(int(i) for i in set(scaffold_kfold_indices(["CCO", "CCN", "c1ccccc1"], 2, 0)[0][1]).union(
...     set(scaffold_kfold_indices(["CCO", "CCN", "c1ccccc1"], 2, 0)[1][1])
... ))
[0, 1, 2]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import (
    AD_THRESHOLDS,
    CALIBRATION_METHOD,
    ENDPOINTS,
    MODELS_DIR,
    N_SPLITS,
    RANDOM_STATE,
)
from featurizer import featurize_batch, is_valid_smiles
from utils import ensure_directories, setup_logging

logger = setup_logging(__name__)

EstimatorFactory = Callable[[float, int], object]


@dataclass
class FoldMetrics:
    endpoint: str
    fold: int
    n_train: int
    n_test: int
    positives_train: int
    positives_test: int
    roc_auc: float
    pr_auc: float
    brier: float
    balanced_accuracy: float
    mcc: float


@dataclass
class MultitaskDataset:
    smiles: List[str]
    X: np.ndarray
    Y: np.ndarray
    endpoints: List[str]


def _default_estimator_factory(scale_pos_weight: float, random_state: int) -> object:
    """Build a default endpoint estimator.

    Imported lazily so module import does not hard-fail when XGBoost
    is unavailable in lightweight environments (for example unit tests).
    """
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError(
            "xgboost is required for training. Install it via requirements.txt"
        ) from exc

    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=float(max(scale_pos_weight, 1.0)),
        verbosity=0,
    )


def generate_scaffold(smiles: str) -> str:
    """Return Bemis-Murcko scaffold or canonical SMILES fallback."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for scaffold generation: {smiles}")

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold else Chem.MolToSmiles(mol, canonical=True)


def scaffold_kfold_indices(
    smiles_list: Sequence[str], n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build scaffold-disjoint K-fold splits.

    Groups compounds by scaffold, then greedily assigns scaffold groups
    to the smallest fold to improve fold-size balance.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold_to_indices[generate_scaffold(smiles)].append(idx)

    groups = list(scaffold_to_indices.values())
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for group in groups:
        fold_idx = int(np.argmin([len(fold) for fold in folds]))
        folds[fold_idx].extend(group)

    all_indices = np.arange(len(smiles_list))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        test_idx = np.array(sorted(folds[fold_idx]), dtype=int)
        train_mask = np.ones(len(smiles_list), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, test_idx))

    return splits


def _canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _normalize_endpoint_columns(
    df: pd.DataFrame, endpoints: Sequence[str]
) -> Tuple[pd.DataFrame, List[str]]:
    columns_lower = {col.lower(): col for col in df.columns}

    if "smiles" not in columns_lower:
        raise ValueError("Dataset must contain a smiles column")

    canonical_endpoints: List[str] = []
    rename_map: Dict[str, str] = {}

    for endpoint in endpoints:
        key = endpoint.lower()
        if key not in columns_lower:
            raise ValueError(
                f"Dataset missing endpoint '{endpoint}'. Available columns: {list(df.columns)}"
            )
        source_col = columns_lower[key]
        rename_map[source_col] = endpoint
        canonical_endpoints.append(endpoint)

    rename_map[columns_lower["smiles"]] = "smiles"
    df = df.rename(columns=rename_map)
    keep_cols = ["smiles", *canonical_endpoints]
    return df[keep_cols].copy(), canonical_endpoints


def load_multitask_dataset(
    csv_path: Path | str, endpoints: Sequence[str] = ENDPOINTS
) -> MultitaskDataset:
    """Load, validate, canonicalize, and featurize multitask training data."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df, endpoint_names = _normalize_endpoint_columns(df, endpoints)

    df["smiles"] = df["smiles"].astype(str).str.strip()
    valid_mask = df["smiles"].apply(is_valid_smiles)
    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        logger.warning("Dropping %s rows with invalid SMILES", invalid_count)
    df = df.loc[valid_mask].copy()

    if df.empty:
        raise ValueError("No valid molecules remain after SMILES validation")

    df["smiles"] = df["smiles"].apply(_canonicalize_smiles)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)
    dropped_dupes = before_dedup - len(df)
    if dropped_dupes:
        logger.info("Dropped %s duplicate canonical SMILES", dropped_dupes)

    for endpoint in endpoint_names:
        df[endpoint] = pd.to_numeric(df[endpoint], errors="coerce")
        unique_vals = sorted(df[endpoint].dropna().unique().tolist())
        if any(val not in (0, 1) for val in unique_vals):
            raise ValueError(
                f"Endpoint '{endpoint}' must be binary (0/1/NaN). Found: {unique_vals[:10]}"
            )

    if all(df[ep].dropna().empty for ep in endpoint_names):
        raise ValueError("All endpoint columns are empty after cleaning")

    X, valid_smiles = featurize_batch(df["smiles"].tolist(), remove_invalid=False)
    if valid_smiles != df["smiles"].tolist():
        raise RuntimeError("Featurizer returned reordered SMILES unexpectedly")

    Y = df[endpoint_names].to_numpy(dtype=float)
    return MultitaskDataset(smiles=valid_smiles, X=X.astype(float), Y=Y, endpoints=list(endpoint_names))


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def _fit_calibrated_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    estimator_factory: EstimatorFactory,
    calibration_folds: int,
    random_state: int,
) -> CalibratedClassifierCV:
    min_class_count = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    n_cal_folds = min(calibration_folds, min_class_count)
    if n_cal_folds < 2:
        raise ValueError("Not enough class support for calibration")

    scale_pos_weight = _compute_scale_pos_weight(y_train)
    estimator = estimator_factory(scale_pos_weight, random_state)

    cal_cv = StratifiedKFold(
        n_splits=n_cal_folds,
        shuffle=True,
        random_state=random_state,
    )

    calibrated = CalibratedClassifierCV(
        estimator,
        method=CALIBRATION_METHOD,
        cv=cal_cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return (
        float(average_precision_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )


def _evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _summarize_fold_metrics(fold_metrics: List[FoldMetrics]) -> Dict[str, Dict[str, float]]:
    """Aggregate fold-level metrics into endpoint-level means/std.

    >>> rows = [
    ...     FoldMetrics("DILI", 1, 10, 5, 4, 2, 0.8, 0.6, 0.2, 0.7, 0.3),
    ...     FoldMetrics("DILI", 2, 11, 6, 5, 2, 0.9, 0.7, 0.1, 0.8, 0.4),
    ... ]
    >>> summary = _summarize_fold_metrics(rows)
    >>> round(summary["DILI"]["roc_auc_mean"], 3)
    0.85
    """
    by_endpoint: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in fold_metrics:
        row_dict = asdict(row)
        endpoint = row.endpoint
        for metric_name in ("roc_auc", "pr_auc", "brier", "balanced_accuracy", "mcc"):
            value = float(row_dict[metric_name])
            if np.isfinite(value):
                by_endpoint[endpoint][metric_name].append(value)

    summary: Dict[str, Dict[str, float]] = {}
    for endpoint, metric_lists in by_endpoint.items():
        endpoint_summary: Dict[str, float] = {"n_folds": float(len(metric_lists.get("roc_auc", [])))}
        for metric_name, values in metric_lists.items():
            if values:
                endpoint_summary[f"{metric_name}_mean"] = float(np.mean(values))
                endpoint_summary[f"{metric_name}_std"] = float(np.std(values, ddof=0))
            else:
                endpoint_summary[f"{metric_name}_mean"] = float("nan")
                endpoint_summary[f"{metric_name}_std"] = float("nan")
        summary[endpoint] = endpoint_summary

    return summary


def _build_training_fingerprints(smiles_list: Sequence[str]) -> List[DataStructs.ExplicitBitVect]:
    fps: List[DataStructs.ExplicitBitVect] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return fps


def cross_validate_models(
    dataset: MultitaskDataset,
    n_splits: int,
    estimator_factory: EstimatorFactory,
    calibration_folds: int,
    random_state: int,
) -> List[FoldMetrics]:
    """Run scaffold-disjoint cross-validation across all endpoints."""
    split_indices = scaffold_kfold_indices(dataset.smiles, n_splits=n_splits, random_state=random_state)

    fold_metrics: List[FoldMetrics] = []
    for fold_id, (train_idx, test_idx) in enumerate(split_indices, start=1):
        X_train_all = dataset.X[train_idx]
        X_test_all = dataset.X[test_idx]

        for endpoint_idx, endpoint in enumerate(dataset.endpoints):
            y_train_all = dataset.Y[train_idx, endpoint_idx]
            y_test_all = dataset.Y[test_idx, endpoint_idx]

            train_mask = ~np.isnan(y_train_all)
            test_mask = ~np.isnan(y_test_all)

            X_train = X_train_all[train_mask]
            y_train = y_train_all[train_mask].astype(int)
            X_test = X_test_all[test_mask]
            y_test = y_test_all[test_mask].astype(int)

            if len(y_train) < 8 or len(y_test) < 4:
                logger.warning(
                    "Skipping fold %s endpoint %s: insufficient labeled data (train=%s, test=%s)",
                    fold_id,
                    endpoint,
                    len(y_train),
                    len(y_test),
                )
                continue

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logger.warning(
                    "Skipping fold %s endpoint %s: single-class split (train classes=%s, test classes=%s)",
                    fold_id,
                    endpoint,
                    np.unique(y_train).tolist(),
                    np.unique(y_test).tolist(),
                )
                continue

            try:
                model = _fit_calibrated_model(
                    X_train=X_train,
                    y_train=y_train,
                    estimator_factory=estimator_factory,
                    calibration_folds=calibration_folds,
                    random_state=random_state + fold_id,
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping fold %s endpoint %s: %s",
                    fold_id,
                    endpoint,
                    exc,
                )
                continue

            y_prob = model.predict_proba(X_test)[:, 1]
            scores = _evaluate_predictions(y_test, y_prob)

            row = FoldMetrics(
                endpoint=endpoint,
                fold=fold_id,
                n_train=int(len(y_train)),
                n_test=int(len(y_test)),
                positives_train=int(np.sum(y_train == 1)),
                positives_test=int(np.sum(y_test == 1)),
                roc_auc=scores["roc_auc"],
                pr_auc=scores["pr_auc"],
                brier=scores["brier"],
                balanced_accuracy=scores["balanced_accuracy"],
                mcc=scores["mcc"],
            )
            fold_metrics.append(row)

        logger.info("Completed scaffold fold %s/%s", fold_id, n_splits)

    return fold_metrics


def fit_final_models(
    dataset: MultitaskDataset,
    estimator_factory: EstimatorFactory,
    calibration_folds: int,
    random_state: int,
) -> Dict[str, CalibratedClassifierCV]:
    """Fit one calibrated classifier per endpoint using all labeled data."""
    models: Dict[str, CalibratedClassifierCV] = {}

    for endpoint_idx, endpoint in enumerate(dataset.endpoints):
        y_all = dataset.Y[:, endpoint_idx]
        mask = ~np.isnan(y_all)

        X_train = dataset.X[mask]
        y_train = y_all[mask].astype(int)

        if len(np.unique(y_train)) < 2:
            logger.warning("Skipping endpoint %s: only one class in full data", endpoint)
            continue

        try:
            model = _fit_calibrated_model(
                X_train=X_train,
                y_train=y_train,
                estimator_factory=estimator_factory,
                calibration_folds=calibration_folds,
                random_state=random_state,
            )
        except ValueError as exc:
            logger.warning(
                "Endpoint %s could not be calibrated (%s). Falling back to uncalibrated estimator.",
                endpoint,
                exc,
            )
            scale_pos_weight = _compute_scale_pos_weight(y_train)
            model = estimator_factory(scale_pos_weight, random_state)
            model.fit(X_train, y_train)

        models[endpoint] = model
        logger.info("Trained final model for %s on %s molecules", endpoint, len(y_train))

    return models


def train_multitask_model(
    dataset_csv: Path | str,
    output_model_path: Optional[Path | str] = None,
    n_splits: int = N_SPLITS,
    calibration_folds: int = 3,
    random_state: int = RANDOM_STATE,
    estimator_factory: Optional[EstimatorFactory] = None,
    save_metrics_json: bool = True,
) -> Dict[str, object]:
    """Train and persist multitask ADMET models.

    Returns a model bundle dictionary that includes models and CV metadata.
    """
    ensure_directories()
    estimator_factory = estimator_factory or _default_estimator_factory

    dataset = load_multitask_dataset(dataset_csv, ENDPOINTS)

    fold_metrics = cross_validate_models(
        dataset=dataset,
        n_splits=n_splits,
        estimator_factory=estimator_factory,
        calibration_folds=calibration_folds,
        random_state=random_state,
    )
    cv_summary = _summarize_fold_metrics(fold_metrics)

    for endpoint, scores in cv_summary.items():
        logger.info(
            "CV %s: ROC-AUC %.3f +/- %.3f | PR-AUC %.3f +/- %.3f",
            endpoint,
            scores.get("roc_auc_mean", float("nan")),
            scores.get("roc_auc_std", float("nan")),
            scores.get("pr_auc_mean", float("nan")),
            scores.get("pr_auc_std", float("nan")),
        )

    final_models = fit_final_models(
        dataset=dataset,
        estimator_factory=estimator_factory,
        calibration_folds=calibration_folds,
        random_state=random_state,
    )

    if not final_models:
        raise RuntimeError("No endpoint models were trained. Check class support per endpoint.")

    training_fps = _build_training_fingerprints(dataset.smiles)

    model_path = Path(output_model_path) if output_model_path else MODELS_DIR / "multitask_admet_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    bundle: Dict[str, object] = {
        "models": final_models,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(Path(dataset_csv)),
            "n_molecules": int(len(dataset.smiles)),
            "feature_dim": int(dataset.X.shape[1]),
            "endpoints": dataset.endpoints,
            "ad_thresholds": dict(AD_THRESHOLDS),
            "n_splits": int(n_splits),
            "calibration_folds": int(calibration_folds),
            "random_state": int(random_state),
            "n_training_fingerprints": int(len(training_fps)),
        },
        "cv_summary": cv_summary,
        "cv_folds": [asdict(row) for row in fold_metrics],
    }

    joblib.dump(bundle, model_path)
    logger.info("Saved multitask model bundle: %s", model_path)

    # Backward compatibility for existing predictor code paths.
    training_fp_path = MODELS_DIR / "training_fingerprints.pkl"
    joblib.dump(training_fps, training_fp_path)
    logger.info("Saved training fingerprints: %s", training_fp_path)

    if save_metrics_json:
        metrics_path = model_path.with_suffix(".metrics.json")
        metrics_payload = {
            "metadata": bundle["metadata"],
            "cv_summary": cv_summary,
            "cv_folds": bundle["cv_folds"],
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        logger.info("Saved metrics JSON: %s", metrics_path)

    return bundle


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train calibrated multitask ADMET models")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).parent / "data" / "admet_multitask.csv"),
        help="Path to multitask CSV dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODELS_DIR / "multitask_admet_model.pkl"),
        help="Output path for the model bundle",
    )
    parser.add_argument("--splits", type=int, default=N_SPLITS, help="Number of scaffold CV folds")
    parser.add_argument(
        "--calibration-folds",
        type=int,
        default=3,
        help="Calibration folds inside each training split",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    train_multitask_model(
        dataset_csv=args.data,
        output_model_path=args.output,
        n_splits=args.splits,
        calibration_folds=args.calibration_folds,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
