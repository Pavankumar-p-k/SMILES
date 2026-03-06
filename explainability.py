"""Explainability helpers for tree-based ADMET endpoint models.

Examples
--------
>>> rows = top_contributing_features(np.array([0.1, -0.4, 0.2]), ["a", "b", "c"], top_n=2)
>>> rows[0]["feature"], round(rows[0]["shap_value"], 1)
('b', -0.4)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from config import DESCRIPTOR_NAMES, FINGERPRINT_BITS, FINGERPRINT_RADIUS
from utils import setup_logging

logger = setup_logging(__name__)

try:
    import shap
except Exception:
    shap = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _unwrap_model(model: Any) -> Any:
    """Extract the underlying tree model from calibrated wrappers."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        calibrated = model.calibrated_classifiers_[0]
        if hasattr(calibrated, "estimator"):
            return calibrated.estimator
        if hasattr(calibrated, "base_estimator"):
            return calibrated.base_estimator

    if hasattr(model, "base_estimator_"):
        return model.base_estimator_

    if hasattr(model, "estimator"):
        return model.estimator

    return model


def _coerce_expected_value(expected_value: Any) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        flat = np.asarray(expected_value).reshape(-1)
        return float(flat[-1])
    return float(expected_value)


def _normalize_feature_names(feature_count: int, feature_names: Optional[Sequence[str]] = None) -> List[str]:
    if feature_names is None:
        feature_names = DESCRIPTOR_NAMES

    names = list(feature_names)
    if len(names) == feature_count:
        return names

    logger.warning(
        "Feature-name length (%s) does not match feature count (%s). Using fallback names.",
        len(names),
        feature_count,
    )
    return [f"feature_{idx}" for idx in range(feature_count)]


def compute_shap_values(model: Any, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute SHAP values for a single sample.

    Notes
    -----
    For calibrated models, SHAP is computed on the wrapped base estimator.
    The calibrated probability transform itself is not decomposed.
    """
    if shap is None:
        raise ImportError("shap is not installed. Install shap to enable explainability.")

    x = np.asarray(X, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    base_model = _unwrap_model(model)
    explainer = shap.TreeExplainer(base_model)

    try:
        shap_output = explainer.shap_values(x, check_additivity=False)
    except TypeError:
        shap_output = explainer.shap_values(x)

    if isinstance(shap_output, list):
        values = np.asarray(shap_output[-1], dtype=float)
    else:
        values = np.asarray(shap_output, dtype=float)
        if values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes)
            values = values[:, :, -1]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    base_value = _coerce_expected_value(explainer.expected_value)
    return values[0], base_value


def plot_shap_waterfall(
    shap_values: np.ndarray,
    features: np.ndarray,
    endpoint: str,
    base_value: float = 0.0,
    feature_names: Optional[Sequence[str]] = None,
    max_display: int = 12,
):
    """Create a SHAP waterfall plot and return a Matplotlib figure."""
    if plt is None:
        raise ImportError("matplotlib is not installed. Install matplotlib to render plots.")
    if shap is None:
        raise ImportError("shap is not installed. Install shap to generate waterfall plots.")

    values = np.asarray(shap_values, dtype=float).reshape(-1)
    feat = np.asarray(features, dtype=float).reshape(-1)
    names = _normalize_feature_names(len(values), feature_names)

    explanation = shap.Explanation(
        values=values,
        base_values=float(base_value),
        data=feat,
        feature_names=names,
    )

    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.title(f"{endpoint} SHAP explanation")
    plt.tight_layout()
    return fig


def top_contributing_features(
    shap_values: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = 10,
) -> List[Dict[str, float | str | int]]:
    """Return top features ranked by absolute SHAP contribution."""
    values = np.asarray(shap_values, dtype=float).reshape(-1)
    names = _normalize_feature_names(len(values), feature_names)

    top_n = max(1, min(int(top_n), len(values)))
    order = np.argsort(np.abs(values))[::-1][:top_n]

    rows: List[Dict[str, float | str | int]] = []
    for idx in order:
        rows.append(
            {
                "index": int(idx),
                "feature": names[idx],
                "shap_value": float(values[idx]),
                "abs_shap": float(abs(values[idx])),
            }
        )
    return rows


def _fingerprint_offset(feature_names: Sequence[str]) -> int:
    for idx, name in enumerate(feature_names):
        if str(name).startswith("Morgan_FP_"):
            return idx
    # Fallback to config convention (7 descriptors first).
    return len(feature_names) - FINGERPRINT_BITS


def extract_top_positive_fingerprint_bits(
    shap_values: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = 6,
) -> List[int]:
    """Return the most positive Morgan fingerprint bits from SHAP values."""
    values = np.asarray(shap_values, dtype=float).reshape(-1)
    names = _normalize_feature_names(len(values), feature_names)
    fp_offset = _fingerprint_offset(names)

    ranked: List[Tuple[int, float]] = []
    for feat_idx in np.argsort(values)[::-1]:
        if values[feat_idx] <= 0:
            break

        name = names[feat_idx]
        if name.startswith("Morgan_FP_"):
            try:
                bit_idx = int(name.replace("Morgan_FP_", ""))
            except ValueError:
                bit_idx = feat_idx - fp_offset
        elif feat_idx >= fp_offset:
            bit_idx = feat_idx - fp_offset
        else:
            continue

        if 0 <= bit_idx < FINGERPRINT_BITS:
            ranked.append((bit_idx, float(values[feat_idx])))

    unique_bits: List[int] = []
    for bit_idx, _ in ranked:
        if bit_idx not in unique_bits:
            unique_bits.append(bit_idx)
        if len(unique_bits) >= max(1, int(top_n)):
            break

    return unique_bits


def morgan_bit_highlights(
    smiles: str,
    bit_indices: Sequence[int],
    radius: int = FINGERPRINT_RADIUS,
    n_bits: int = FINGERPRINT_BITS,
) -> Dict[str, Any]:
    """Map Morgan bits to atom and bond indices for highlighting."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)

    atom_indices = set()
    bond_indices = set()
    bits_found: List[int] = []

    for bit in bit_indices:
        if bit not in bit_info:
            continue
        bits_found.append(int(bit))
        for atom_idx, rad in bit_info[bit]:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            atom_indices.add(atom_idx)
            for bond_idx in env:
                bond_indices.add(bond_idx)
                bond = mol.GetBondWithIdx(bond_idx)
                atom_indices.add(bond.GetBeginAtomIdx())
                atom_indices.add(bond.GetEndAtomIdx())

    return {
        "mol": mol,
        "bits_found": bits_found,
        "atom_indices": sorted(atom_indices),
        "bond_indices": sorted(bond_indices),
    }


def draw_highlighted_substructures(
    smiles: str,
    bit_indices: Sequence[int],
    size: Tuple[int, int] = (480, 320),
):
    """Draw the molecule with SHAP-prioritized fingerprint substructures highlighted."""
    mapping = morgan_bit_highlights(smiles, bit_indices)
    return Draw.MolToImage(
        mapping["mol"],
        size=size,
        highlightAtoms=mapping["atom_indices"],
        highlightBonds=mapping["bond_indices"],
    )
