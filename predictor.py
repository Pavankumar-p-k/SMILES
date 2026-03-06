"""Prediction utilities for the In-Silico ADMET Risk Profiler.

Examples
--------
>>> classify_domain(0.75)
'In-Domain'
>>> classify_domain(0.52)
'Near-Boundary'
>>> classify_domain(0.12)
'Out-of-Domain'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from config import AD_THRESHOLDS, ENDPOINTS, FINGERPRINT_BITS, FINGERPRINT_RADIUS, MODELS_DIR
from featurizer import featurize_smiles, is_valid_smiles
from utils import setup_logging

logger = setup_logging(__name__)


def classify_domain(max_similarity: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Map maximum Tanimoto similarity to AD status."""
    thresholds = thresholds or AD_THRESHOLDS
    if max_similarity >= float(thresholds.get("In-Domain", 0.7)):
        return "In-Domain"
    if max_similarity >= float(thresholds.get("Near-Boundary", 0.5)):
        return "Near-Boundary"
    return "Out-of-Domain"


class ADMETPredictor:
    """Load trained models and run endpoint predictions."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        training_fp_path: Optional[Path | str] = None,
        models_dir: Path | str = MODELS_DIR,
        enable_explainability: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.model_path = Path(model_path) if model_path else self.models_dir / "multitask_admet_model.pkl"
        self.training_fp_path = (
            Path(training_fp_path) if training_fp_path else self.models_dir / "training_fingerprints.pkl"
        )
        self.enable_explainability = enable_explainability

        self.model_bundle: Dict[str, Any] = {}
        self.models: Dict[str, Any] = self._load_models()
        self.training_fps: List[DataStructs.ExplicitBitVect] = self._load_training_fingerprints()

    def _load_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {}

        if self.model_path.exists():
            artifact = joblib.load(self.model_path)
            if isinstance(artifact, dict) and "models" in artifact and isinstance(artifact["models"], dict):
                self.model_bundle = artifact
                models.update(artifact["models"])
            elif isinstance(artifact, dict):
                # Backward-compatible shape: endpoint -> estimator.
                models.update(artifact)
            else:
                raise ValueError(
                    f"Unsupported model artifact format in {self.model_path}. "
                    "Expected dict or bundle with 'models' key."
                )

        if not models:
            # Backward-compatible individual endpoint model files.
            for endpoint in ENDPOINTS:
                candidate = self.models_dir / f"{endpoint}_model.pkl"
                if candidate.exists():
                    models[endpoint] = joblib.load(candidate)

        if not models:
            raise FileNotFoundError(
                "No models found. Expected either a bundled multitask model or per-endpoint model files "
                f"under {self.models_dir}."
            )

        invalid = [endpoint for endpoint, model in models.items() if not hasattr(model, "predict_proba")]
        if invalid:
            raise ValueError(f"Loaded models without predict_proba support: {invalid}")

        logger.info("Loaded %s endpoint model(s): %s", len(models), sorted(models.keys()))
        return models

    def _load_training_fingerprints(self) -> List[DataStructs.ExplicitBitVect]:
        if self.training_fp_path.exists():
            fps = joblib.load(self.training_fp_path)
            if isinstance(fps, list):
                logger.info("Loaded %s training fingerprints", len(fps))
                return fps
            logger.warning("Training fingerprint file has unexpected format: %s", self.training_fp_path)

        logger.warning(
            "Training fingerprints not found at %s. Applicability-domain status will be 'Unknown'.",
            self.training_fp_path,
        )
        return []

    def compute_ad(self, smiles: str) -> Tuple[str, Optional[float]]:
        """Compute applicability-domain status using max Tanimoto similarity."""
        if not self.training_fps:
            return "Unknown", None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        query_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=FINGERPRINT_RADIUS,
            nBits=FINGERPRINT_BITS,
        )

        max_similarity = max(
            float(DataStructs.TanimotoSimilarity(query_fp, train_fp))
            for train_fp in self.training_fps
        )
        return classify_domain(max_similarity), max_similarity

    def _predict_probabilities(self, features: np.ndarray) -> Dict[str, float]:
        preds: Dict[str, float] = {}
        for endpoint, model in self.models.items():
            prob = float(model.predict_proba(features.reshape(1, -1))[0][1])
            preds[endpoint] = prob
        return preds

    def predict_single(
        self,
        smiles: str,
        include_shap: bool = False,
        top_k_features: int = 12,
    ) -> Dict[str, Any]:
        """Predict all endpoints for one SMILES string.

        Returns a structured dictionary suitable for both API and UI layers.
        """
        if not is_valid_smiles(smiles):
            return {
                "valid": False,
                "smiles": smiles,
                "error": "Invalid SMILES",
                "predictions": {},
                "ad_status": "Unknown",
                "ad_similarity": None,
            }

        try:
            features = featurize_smiles(smiles).astype(float)
            predictions = self._predict_probabilities(features)
            ad_status, ad_similarity = self.compute_ad(smiles)

            result: Dict[str, Any] = {
                "valid": True,
                "smiles": smiles,
                "predictions": predictions,
                "ad_status": ad_status,
                "ad_similarity": ad_similarity,
            }

            if include_shap and self.enable_explainability:
                try:
                    from explainability import compute_shap_values, top_contributing_features

                    shap_payload: Dict[str, Any] = {}
                    for endpoint, model in self.models.items():
                        shap_values, base_value = compute_shap_values(model=model, X=features)
                        shap_payload[endpoint] = {
                            "base_value": float(base_value),
                            "shap_values": shap_values.tolist(),
                            "top_features": top_contributing_features(shap_values, top_n=top_k_features),
                            "features": features.tolist(),
                        }
                    result["shap"] = shap_payload
                except Exception as shap_exc:
                    logger.warning("SHAP generation skipped for %s: %s", smiles, shap_exc)
                    result["shap_error"] = str(shap_exc)

            return result

        except Exception as exc:
            logger.exception("Prediction failed for %s", smiles)
            return {
                "valid": False,
                "smiles": smiles,
                "error": str(exc),
                "predictions": {},
                "ad_status": "Unknown",
                "ad_similarity": None,
            }

    def predict_batch(self, smiles_list: List[str], include_shap: bool = False) -> List[Dict[str, Any]]:
        """Predict endpoints for a batch of SMILES strings."""
        return [self.predict_single(smiles, include_shap=include_shap) for smiles in smiles_list]
