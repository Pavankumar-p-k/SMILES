"""Unified holographic Streamlit dashboard for ADMET endpoint prediction."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from config import ENDPOINT_INFO
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

st.set_page_config(page_title="HOLOSCAN ADMET", layout="wide")

TEMPLATE_PATH = Path(__file__).parent / "ui" / "HoloAnatomyLab.html"


def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.3:
        return "MODERATE"
    return "LOW"


def _risk_color(prob: float) -> str:
    if prob >= 0.7:
        return "#ff4d6d"
    if prob >= 0.3:
        return "#ffcc00"
    return "#00ff88"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url("https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap");
            .stApp {
                background:
                    radial-gradient(circle at 12% 18%, rgba(0,245,255,0.10), transparent 34%),
                    radial-gradient(circle at 85% 8%, rgba(0,80,255,0.14), transparent 36%),
                    #020b18;
                color: #00f5ff;
                font-family: "Share Tech Mono", monospace;
            }
            .main .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1.2rem;
                max-width: 96%;
            }
            h1, h2, h3, h4, p, label, span, div {
                font-family: "Share Tech Mono", monospace;
            }
            .stButton > button,
            .stDownloadButton > button {
                border: 1px solid rgba(0,245,255,0.45) !important;
                background: rgba(0,245,255,0.09) !important;
                color: #00f5ff !important;
                letter-spacing: 0.06em;
            }
            .stButton > button:hover,
            .stDownloadButton > button:hover {
                border-color: #00f5ff !important;
                box-shadow: 0 0 12px rgba(0,245,255,0.35) !important;
            }
            .stTextInput input,
            .stSelectbox div[data-baseweb="select"] > div,
            .stTextArea textarea {
                background: rgba(0,25,45,0.86) !important;
                border: 1px solid rgba(0,245,255,0.28) !important;
                color: #d7fbff !important;
            }
            .stTabs [data-baseweb="tab"] {
                border: 1px solid rgba(0,245,255,0.25);
                background: rgba(0,245,255,0.06);
                border-radius: 0;
            }
            .stTabs [aria-selected="true"] {
                background: rgba(0,245,255,0.18) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _draw_molecule_2d(smiles: str, size=(420, 300)) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Could not render 2D molecule")
        return
    st.image(Draw.MolToImage(mol, size=size), use_container_width=False)


def _smiles_to_3d_molblock(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol)
        if status != 0:
            return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToMolBlock(mol)


def _render_3d_molecule(smiles: str) -> None:
    mol_block = _smiles_to_3d_molblock(smiles)
    if mol_block is None:
        st.warning("3D molecular diagram unavailable for this SMILES.")
        return

    html = f"""
    <html>
    <head>
      <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
      <style>
        body {{ margin:0; background:#020b18; }}
        #viewer {{ width:100%; height:420px; border:1px solid rgba(0,245,255,0.28); }}
      </style>
    </head>
    <body>
      <div id="viewer"></div>
      <script>
        const mol = {json.dumps(mol_block)};
        const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "#020b18" }});
        viewer.addModel(mol, "mol");
        viewer.setStyle({{}}, {{stick: {{radius: 0.18, colorscheme: "cyanCarbon"}}, sphere: {{scale: 0.25}}}});
        viewer.zoomTo();
        viewer.spin(true);
        viewer.render();
      </script>
    </body>
    </html>
    """
    components.html(html, height=430, scrolling=False)


def _build_organ_panel(predictions: Dict[str, float]) -> Dict[str, Dict[str, str]]:
    bbb = float(predictions.get("BBB", 0.0))
    dili = float(predictions.get("DILI", 0.0))
    herg = float(predictions.get("hERG", 0.0))
    ames = float(predictions.get("Ames", 0.0))

    def build(prob: float, msg_hi: str, msg_lo: str) -> Dict[str, str]:
        pct = prob * 100.0
        return {
            "label": _risk_label(prob),
            "score": f"{prob:.3f}",
            "pct": f"{pct:.1f}%",
            "bar": f"{pct:.1f}",
            "color": _risk_color(prob),
            "opacity": f"{0.16 + (0.62 * prob):.2f}",
            "message": msg_hi if prob >= 0.7 else msg_lo,
        }

    return {
        "brain": build(bbb, "BBB penetration toxicity elevated.", "BBB risk currently controlled."),
        "heart": build(herg, "hERG cardiac risk elevated.", "hERG risk currently controlled."),
        "liver": build(dili, "DILI hepatotoxicity elevated.", "DILI risk currently controlled."),
        "genetic": build(ames, "Ames mutagenicity elevated.", "Mutagenicity risk currently controlled."),
    }


def _risk_flags(predictions: Dict[str, float]) -> List[str]:
    flags: List[str] = []
    bbb = float(predictions.get("BBB", 0.0))
    dili = float(predictions.get("DILI", 0.0))
    herg = float(predictions.get("hERG", 0.0))
    ames = float(predictions.get("Ames", 0.0))

    if bbb >= 0.7:
        flags.append("Brain risk HIGH: BBB toxicity signal elevated.")
    elif bbb >= 0.3:
        flags.append("Brain risk MODERATE: BBB signal needs follow-up.")
    else:
        flags.append("Brain risk LOW by BBB model output.")

    if dili >= 0.7:
        flags.append("Liver risk HIGH: DILI output elevated.")
    if herg >= 0.7:
        flags.append("Cardiac risk HIGH: hERG output elevated.")
    if ames >= 0.7:
        flags.append("Genetic risk HIGH: Ames output elevated.")

    if len(flags) == 1:
        flags.append("No additional organ in high-risk zone.")
    return flags


def _render_dashboard_html(smiles: str, result: Dict[str, object] | None) -> str:
    predictions = {}
    ad_status = "Unknown"
    ad_similarity = None
    recommendation = "Run prediction to generate recommendation."
    risk_category = "N/A"
    composite_score = 0.0

    if result and result.get("valid"):
        predictions = result.get("predictions", {})
        summary = get_risk_summary(predictions)
        recommendation = str(summary["recommendation"])
        risk_category = str(summary["risk_category"])
        composite_score = float(summary["composite_score"])
        ad_status = str(result.get("ad_status", "Unknown"))
        ad_similarity = result.get("ad_similarity")

    organ = _build_organ_panel(predictions)
    flags = _risk_flags(predictions)
    flags_html = "".join(f'<div class="flag">{escape(item)}</div>' for item in flags)
    ad_similarity_text = (
        f"Max Tanimoto similarity: {float(ad_similarity):.3f}" if ad_similarity is not None else "Similarity unavailable"
    )

    if not TEMPLATE_PATH.exists():
        return "<html><body style='background:#020b18;color:#00f5ff;'>Template missing.</body></html>"

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__RISK_COLOR__": "#00ff88" if risk_category == "Low" else "#ffcc00" if risk_category == "Moderate" else "#ff4d6d",
        "__COMPOSITE_SCORE__": f"{composite_score:.3f}",
        "__RISK_CATEGORY__": risk_category,
        "__SMILES_TEXT__": escape(smiles if smiles else "N/A"),
        "__AD_STATUS__": escape(ad_status),
        "__AD_STATUS_DETAIL__": escape(ad_status if ad_similarity is None else f"{ad_status} ({float(ad_similarity):.3f})"),
        "__AD_SIMILARITY__": escape(ad_similarity_text),
        "__RECOMMENDATION__": escape(recommendation),
        "__RISK_FLAGS_HTML__": flags_html,

        "__BRAIN_COLOR__": organ["brain"]["color"],
        "__HEART_COLOR__": organ["heart"]["color"],
        "__LIVER_COLOR__": organ["liver"]["color"],
        "__GEN_COLOR__": organ["genetic"]["color"],

        "__BBB_SCORE__": organ["brain"]["score"],
        "__HERG_SCORE__": organ["heart"]["score"],
        "__DILI_SCORE__": organ["liver"]["score"],
        "__AMES_SCORE__": organ["genetic"]["score"],

        "__BBB_PCT__": organ["brain"]["pct"],
        "__HERG_PCT__": organ["heart"]["pct"],
        "__DILI_PCT__": organ["liver"]["pct"],
        "__AMES_PCT__": organ["genetic"]["pct"],

        "__BBB_BAR__": organ["brain"]["bar"],
        "__HERG_BAR__": organ["heart"]["bar"],
        "__DILI_BAR__": organ["liver"]["bar"],
        "__AMES_BAR__": organ["genetic"]["bar"],

        "__BRAIN_OPACITY__": organ["brain"]["opacity"],
        "__HEART_OPACITY__": organ["heart"]["opacity"],
        "__LIVER_OPACITY__": organ["liver"]["opacity"],
        "__GEN_OPACITY__": organ["genetic"]["opacity"],
    }

    for k, v in replacements.items():
        template = template.replace(k, v)
    return template


@st.cache_resource(show_spinner=False)
def _load_predictor() -> ADMETPredictor:
    return ADMETPredictor()


def _run_single_mode(predictor: ADMETPredictor) -> None:
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "active_smiles" not in st.session_state:
        st.session_state.active_smiles = "CCO"

    st.markdown("### Single Molecule")
    with st.form("predict_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([2.2, 1.1, 1.1])
        with c1:
            smiles_input = st.text_input("SMILES", value=st.session_state.active_smiles, key="smiles_input")
        with c2:
            compute_explainability = st.checkbox("SHAP", value=True)
        with c3:
            run = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    if run:
        st.session_state.active_smiles = smiles_input
        st.session_state.single_result = predictor.predict_single(
            smiles_input,
            include_shap=compute_explainability,
        )

    result = st.session_state.single_result
    dashboard_html = _render_dashboard_html(st.session_state.active_smiles, result)
    components.html(dashboard_html, height=760, scrolling=False)

    if result and not result.get("valid", False):
        st.error(result.get("error", "Prediction failed"))
        return
    if not (result and result.get("valid", False)):
        st.info("Run a prediction to populate live model percentages, organ-risk panels, and molecule diagrams.")
        return

    predictions = result["predictions"]
    summary = get_risk_summary(predictions)

    st.markdown("### Molecular Diagrams")
    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown("#### 2D Molecule")
        _draw_molecule_2d(result["smiles"])
    with d2:
        st.markdown("#### 3D Molecular Diagram")
        _render_3d_molecule(result["smiles"])

    tab_risk, tab_models, tab_explain, tab_export = st.tabs(
        ["Risk Graph", "Model Outputs", "Explainability", "Export"]
    )

    with tab_risk:
        graph_df = pd.DataFrame(
            [{"Endpoint": ep, "Probability": float(prob)} for ep, prob in sorted(predictions.items())]
        )
        st.bar_chart(graph_df.set_index("Endpoint"))

    with tab_models:
        rows = []
        for endpoint, prob in sorted(predictions.items()):
            rows.append(
                {
                    "Organ": {
                        "BBB": "Brain",
                        "DILI": "Liver",
                        "Ames": "Genetic",
                        "hERG": "Heart",
                    }.get(endpoint, endpoint),
                    "Endpoint": endpoint,
                    "Probability": f"{prob:.3f}",
                    "Percent": f"{prob * 100:.1f}%",
                    "Risk": _risk_label(float(prob)),
                    "Model": ENDPOINT_INFO.get(endpoint, {}).get("type", "Unknown"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_explain:
        shap_block = result.get("shap")
        if result.get("shap_error"):
            st.warning(f"Explainability unavailable: {result['shap_error']}")
        if not shap_block:
            st.caption("Enable SHAP and rerun prediction to see explainability.")
        else:
            endpoint_to_explain = st.selectbox("Endpoint", sorted(shap_block.keys()))
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

            st.caption("Top SHAP contributions")
            st.dataframe(pd.DataFrame(endpoint_payload["top_features"]), use_container_width=True)

            bit_indices = extract_top_positive_fingerprint_bits(shap_values, top_n=6)
            if bit_indices:
                st.caption("Highlighted substructures from top positive fingerprint bits")
                highlighted = draw_highlighted_substructures(result["smiles"], bit_indices)
                st.image(highlighted, use_container_width=False)

    with tab_export:
        st.metric("Composite Risk", f"{summary['composite_score']:.3f}", str(summary["risk_category"]))
        if st.button("Generate PDF Report"):
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


def _run_batch_mode(predictor: ADMETPredictor) -> None:
    st.markdown("### Batch CSV")
    uploaded = st.file_uploader("Upload CSV with a 'smiles' column", type=["csv"])
    run_batch = st.button("Run Batch Prediction", type="primary")

    if not run_batch:
        return
    if uploaded is None:
        st.error("Upload a CSV file first")
        return

    df = pd.read_csv(uploaded)
    if "smiles" not in df.columns:
        st.error("CSV must contain a 'smiles' column")
        return

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
    st.dataframe(out_df, use_container_width=True)
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="admet_batch_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    _inject_styles()
    st.markdown("## HOLOSCAN ADMET Command Dashboard")
    st.caption("Single professional dashboard with live organ-linked model outputs and 3D molecular diagrams.")

    try:
        predictor = _load_predictor()
    except Exception as exc:
        st.error(f"Failed to initialize predictor: {exc}")
        st.stop()

    mode = st.radio("Operation Mode", ["Single molecule", "Batch CSV"], horizontal=True)
    if mode == "Single molecule":
        _run_single_mode(predictor)
    else:
        _run_batch_mode(predictor)

    st.caption(
        "Disclaimer: model outputs are for screening support only and do not replace experimental validation."
    )


if __name__ == "__main__":
    main()
