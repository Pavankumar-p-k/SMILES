"""Risk aggregation utilities for endpoint predictions.

Examples
--------
>>> score, category = compute_composite_risk({"DILI": 0.8, "BBB": 0.2, "Ames": 0.1, "hERG": 0.7})
>>> 0.0 <= score <= 1.0
True
>>> category in {"Low", "Moderate", "High"}
True
"""

from __future__ import annotations

from typing import Dict, Tuple

from config import ENDPOINT_WEIGHTS, RISK_CATEGORIES


def classify_risk(score: float) -> str:
    """Map a score in [0, 1] to a categorical risk label."""
    for category, (low, high) in RISK_CATEGORIES.items():
        if low <= score < high:
            return category
    return "High"


def compute_composite_risk(predictions: Dict[str, float]) -> Tuple[float, str]:
    """Compute weighted composite risk with multi-endpoint amplification."""
    if not predictions:
        return 0.0, "Low"

    weighted_sum = 0.0
    total_weight = 0.0

    for endpoint, weight in ENDPOINT_WEIGHTS.items():
        if endpoint in predictions:
            weighted_sum += float(predictions[endpoint]) * weight
            total_weight += weight

    if total_weight <= 0:
        return 0.0, "Low"

    base_score = weighted_sum / total_weight

    # Amplify when multiple endpoints exceed high-risk probability.
    high_count = sum(1 for prob in predictions.values() if prob >= 0.7)
    amplification = 1.0 + (0.05 * high_count)

    composite_score = max(0.0, min(base_score * amplification, 1.0))
    return composite_score, classify_risk(composite_score)


def get_risk_summary(predictions: Dict[str, float]) -> Dict[str, object]:
    """Return a structured risk summary for UI/reporting layers."""
    composite_score, category = compute_composite_risk(predictions)

    endpoint_labels = {
        endpoint: ("High" if prob >= 0.7 else "Moderate" if prob >= 0.3 else "Low")
        for endpoint, prob in predictions.items()
    }

    recommendation_map = {
        "Low": "Proceed with routine screening; risk profile is acceptable.",
        "Moderate": "Proceed with caution and prioritize orthogonal confirmatory assays.",
        "High": "Prioritize experimental validation before progression decisions.",
    }

    return {
        "composite_score": composite_score,
        "risk_category": category,
        "endpoint_categories": endpoint_labels,
        "recommendation": recommendation_map[category],
    }
