"""Streamlit UI for ADMET endpoint prediction, AD checks, and explainability."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from explainability import (
    draw_highlighted_substructures,
    extract_top_positive_fingerprint_bits,
    plot_shap_waterfall,
)
from predictor import ADMETPredictor
from report_generator import generate_report
from risk_aggregator import get_risk_summary
from utils import ensure_directories, setup_logging

logger = setup_logging(__name__, log_file=False)
ensure_directories()

st.set_page_config(page_title="In-Silico ADMET Risk Profiler", layout="wide")

st.title("In-Silico ADMET Risk Profiler")
st.caption(
    "Calibrated endpoint screening for DILI, BBB, Ames, and hERG with applicability-domain checks."
)


def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.3:
        return "Moderate"
    return "Low"


def _draw_molecule(smiles: str, size=(340, 260)) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Could not render molecule")
        return
    st.image(Draw.MolToImage(mol, size=size), use_container_width=False)


@st.cache_resource(show_spinner=False)
def _load_predictor() -> ADMETPredictor:
    return ADMETPredictor()


try:
    predictor = _load_predictor()
except Exception as exc:
    st.error(f"Failed to initialize predictor: {exc}")
    st.stop()

if "single_result" not in st.session_state:
    st.session_state.single_result = None

st.sidebar.header("Input")
mode = st.sidebar.radio("Mode", ["Single molecule", "Batch CSV"])


if mode == "Single molecule":
    smiles_input = st.sidebar.text_input("SMILES", value="CCO")
    compute_explainability = st.sidebar.checkbox("Compute SHAP explanation", value=True)

    predict_clicked = st.sidebar.button("Run prediction", type="primary")
    if predict_clicked:
        result = predictor.predict_single(smiles_input, include_shap=compute_explainability)
        st.session_state.single_result = result

    result = st.session_state.single_result

    if result:
        if not result.get("valid", False):
            st.error(result.get("error", "Prediction failed"))
        else:
            predictions: Dict[str, float] = result["predictions"]
            summary = get_risk_summary(predictions)

            left, right = st.columns([1, 2])
            with left:
                st.subheader("Structure")
                _draw_molecule(result["smiles"])

            with right:
                st.subheader("Endpoint risks")
                endpoint_cols = st.columns(max(1, len(predictions)))
                for (endpoint, prob), col in zip(sorted(predictions.items()), endpoint_cols):
                    with col:
                        st.metric(endpoint, f"{prob:.3f}", _risk_label(prob))

                st.divider()
                st.metric("Composite risk", f"{summary['composite_score']:.3f}", summary["risk_category"])
                st.info(summary["recommendation"])

                ad_status = result.get("ad_status", "Unknown")
                ad_similarity = result.get("ad_similarity")
                if ad_similarity is None:
                    st.caption(f"Applicability domain: {ad_status}")
                else:
                    st.caption(
                        f"Applicability domain: {ad_status} (max Tanimoto similarity {ad_similarity:.3f})"
                    )

            shap_block = result.get("shap")
            if result.get("shap_error"):
                st.warning(f"Explainability unavailable: {result['shap_error']}")
            if shap_block:
                st.divider()
                st.subheader("Explainability")

                endpoint_to_explain = st.selectbox(
                    "Endpoint",
                    sorted(shap_block.keys()),
                )
                endpoint_payload = shap_block[endpoint_to_explain]

                shap_values = np.asarray(endpoint_payload["shap_values"], dtype=float)
                features = np.asarray(endpoint_payload["features"], dtype=float)
                base_value = float(endpoint_payload["base_value"])

                fig = plot_shap_waterfall(
                    shap_values=shap_values,
                    features=features,
                    endpoint=endpoint_to_explain,
                    base_value=base_value,
                    max_display=12,
                )
                st.pyplot(fig, clear_figure=True)

                top_df = pd.DataFrame(endpoint_payload["top_features"])
                st.caption("Top SHAP contributions")
                st.dataframe(top_df, use_container_width=True)

                bit_indices = extract_top_positive_fingerprint_bits(shap_values, top_n=6)
                if bit_indices:
                    st.caption("Highlighted substructures from top positive fingerprint bits")
                    highlighted = draw_highlighted_substructures(result["smiles"], bit_indices)
                    st.image(highlighted, use_container_width=False)

            st.divider()
            if st.button("Generate PDF report"):
                report_path = generate_report(
                    smiles=result["smiles"],
                    predictions=predictions,
                    composite_score=summary["composite_score"],
                    risk_category=summary["risk_category"],
                    ad_status=result.get("ad_status"),
                    ad_similarity=result.get("ad_similarity"),
                )
                if report_path:
                    st.success(f"Report generated: {report_path.name}")
                    with open(report_path, "rb") as handle:
                        st.download_button(
                            label="Download report",
                            data=handle,
                            file_name=report_path.name,
                            mime="application/pdf",
                        )


else:
    uploaded = st.sidebar.file_uploader("Upload CSV with a 'smiles' column", type=["csv"])
    run_batch = st.sidebar.button("Run batch", type="primary")

    if run_batch:
        if uploaded is None:
            st.error("Upload a CSV file first")
        else:
            df = pd.read_csv(uploaded)
            if "smiles" not in df.columns:
                st.error("CSV must contain a 'smiles' column")
            else:
                smiles_series = df["smiles"].astype(str).tolist()
                batch_results = predictor.predict_batch(smiles_series, include_shap=False)

                rows = []
                for item in batch_results:
                    row = {
                        "smiles": item.get("smiles"),
                        "valid": item.get("valid", False),
                        "ad_status": item.get("ad_status"),
                        "ad_similarity": item.get("ad_similarity"),
                        "error": item.get("error"),
                    }
                    for endpoint, prob in item.get("predictions", {}).items():
                        row[endpoint] = prob
                    if item.get("predictions"):
                        risk = get_risk_summary(item["predictions"])
                        row["composite_score"] = risk["composite_score"]
                        row["risk_category"] = risk["risk_category"]
                    rows.append(row)

                out_df = pd.DataFrame(rows)
                st.subheader("Batch predictions")
                st.dataframe(out_df, use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="admet_batch_predictions.csv",
                    mime="text/csv",
                )


st.divider()
st.caption(
    "Disclaimer: model outputs are for screening support only and do not replace experimental validation."
)
