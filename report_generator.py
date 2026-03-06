"""PDF report generation."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import io

from rdkit import Chem
from rdkit.Chem import Draw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from config import REPORTS_DIR, RISK_COLORS, ENDPOINT_INFO
from utils import setup_logging

logger = setup_logging(__name__)


def _draw_molecule(smiles: str) -> Optional[io.BytesIO]:
    """Draw molecule structure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(300, 300))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        return img_buffer
    except Exception as e:
        logger.error(f"Error drawing molecule: {e}")
        return None


def _create_risk_table(
    predictions: Dict[str, float], composite: float, category: str
) -> Table:
    """Create risk summary table."""
    data = [["Endpoint", "Probability", "Risk"]]
    for endpoint, prob in predictions.items():
        risk = "High" if prob >= 0.7 else "Moderate" if prob >= 0.3 else "Low"
        data.append([endpoint, f"{prob:.3f}", risk])

    data.append(["COMPOSITE", f"{composite:.3f}", category])

    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), TA_CENTER),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    return table


def generate_report(
    smiles: str,
    predictions: Dict[str, float],
    composite_score: float,
    risk_category: str,
    ad_status: Optional[str] = None,
    ad_similarity: Optional[float] = None,
) -> Optional[Path]:
    """
    Generate comprehensive PDF screening report.

    Args:
        smiles: SMILES string
        predictions: Dict mapping endpoint → probability
        composite_score: Composite risk score
        risk_category: Risk category (Low, Moderate, High)
        ad_status: Applicability domain status
        ad_similarity: Max Tanimoto similarity

    Returns:
        Path to PDF or None on error
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"report_{timestamp}.pdf"

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#003366"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )

        elements = []
        elements.append(Paragraph("ADMET Screening Report", title_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Molecule section
        elements.append(Paragraph("Molecular Structure", styles["Heading2"]))
        img_buffer = _draw_molecule(smiles)
        if img_buffer:
            img = Image(img_buffer, width=2 * inch, height=2 * inch)
            elements.append(img)
        elements.append(Spacer(1, 0.2 * inch))

        # Risk summary
        elements.append(Paragraph("Risk Assessment", styles["Heading2"]))
        risk_table = _create_risk_table(predictions, composite_score, risk_category)
        elements.append(risk_table)
        elements.append(Spacer(1, 0.2 * inch))

        # AD assessment
        if ad_status:
            elements.append(Paragraph("Applicability Domain", styles["Heading2"]))
            ad_text = f"Status: {ad_status}"
            if ad_similarity is not None:
                ad_text += f" (Tanimoto: {ad_similarity:.3f})"
            elements.append(Paragraph(ad_text, styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))

        # Disclaimer
        elements.append(Spacer(1, 0.2 * inch))
        disclaimer = (
            "These predictions are for early-stage screening only. "
            "Rigorous experimental validation is essential before therapeutic development."
        )
        elements.append(
            Paragraph(
                disclaimer,
                ParagraphStyle(
                    "Disclaimer",
                    parent=styles["Normal"],
                    fontSize=8,
                    textColor=colors.grey,
                    alignment=TA_JUSTIFY,
                ),
            )
        )

        doc.build(elements)
        logger.info(f"Report saved: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None
