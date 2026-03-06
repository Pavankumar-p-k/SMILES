# In-Silico Multi-Endpoint ADMET Risk Profiler

A production-grade machine learning system for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) hazards in drug discovery.

## Features

ðŸ”¬ **Multi-Endpoint ADMET Screening**
- Drug-Induced Liver Injury (DILI) - XGBoost
- Blood-Brain Barrier (BBB) Penetration - XGBoost
- Ames Mutagenicity - Random Forest
- hERG Cardiac Toxicity - Random Forest

ðŸ“Š **Advanced Chemoinformatics**
- RDKit Morgan fingerprints (2048-bit)
- Physicochemical descriptors (MW, LogP, HBD, HBA, TPSA, rotatable bonds, aromatic rings)
- Applicability domain assessment via Tanimoto similarity
- Lipinski Rule of Five evaluation

ðŸ“ˆ **Rigorous ML Pipeline**
- 5-fold stratified cross-validation
- Class imbalance handling (scale_pos_weight, class_weight="balanced")
- Probability outputs for confidence scoring
- Evaluation metrics: ROC-AUC, Accuracy, Precision, Recall

ðŸŽ¨ **Interactive Web Dashboard**
- Streamlit-based UI for single and batch predictions
- 2D molecular structure visualization
- Color-coded risk indicators (ðŸŸ¢ Low, ðŸŸ¡ Moderate, ðŸ”´ High)
- CSV upload/download for batch screening

ðŸ“„ **Scientific Report Generation**
- PDF reports with molecular structures
- Risk interpretation and recommendations
- Drug-likeness assessment
- Applicability domain analysis

## Installation

### Requirements
- Python 3.11+
- NumPy <2.0 (RDKit compatibility)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/in-silico-admet.git
cd in_silico_admet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Models

```bash
python train_multitask.py --data data/admet_multitask.csv
```

This trains calibrated endpoint models for `DILI`, `BBB`, `Ames`, and `hERG`.
Artifacts are saved to:
- `models/multitask_admet_model.pkl`
- `models/training_fingerprints.pkl`
### 2. Web Dashboard

```bash
streamlit run app.py
```

Access at `http://localhost:8501`

### 3. Generate Reports

```python
from report_generator import generate_report
from risk_aggregator import get_risk_summary

predictions = {
    "DILI": 0.15,
    "BBB": 0.82,
    "Ames": 0.45,
    "hERG": 0.38,
}

summary = get_risk_summary(predictions)

report_path = generate_report(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    predictions=predictions,
    composite_score=summary["composite_score"],
    risk_category=summary["risk_category"],
    ad_status="In-Domain",
    ad_similarity=0.72,
)
```
### 4. Python API

```python
from predictor import ADMETPredictor
from lipinski import evaluate_lipinski

predictor = ADMETPredictor()

# Predict single molecule
smiles = "CCO"
result = predictor.predict_single(smiles)
print(result["predictions"])
print(result["ad_status"], result["ad_similarity"])

# Check drug-likeness
passes, violations, details = evaluate_lipinski(smiles)
```
## Project Structure

```
in_silico_admet/
â”œâ”€â”€ train_models.py              # Model training pipeline
â”œâ”€â”€ applicability_domain.py      # Chemical space assessment
â”œâ”€â”€ lipinski.py                  # Drug-likeness evaluation
â”œâ”€â”€ report_generator.py          # PDF report generation
â”œâ”€â”€ app.py                       # Streamlit web dashboard
â”œâ”€â”€ models/                      # Trained model storage
â”œâ”€â”€ reports/                     # Generated PDF reports
â”œâ”€â”€ data/                        # Training datasets
â”œâ”€â”€ requirements.txt             # Python dependencies
â””â”€â”€ README.md                    # This file
```

## ML Architecture

### Feature Engineering (2055 dimensions)
```
RDKit Descriptors (7):
  - Molecular Weight
  - LogP (partition coefficient)
  - H-Bond Donors
  - H-Bond Acceptors
  - Rotatable Bonds
  - TPSA (topological polar surface area)
  - Aromatic Rings

+ Morgan Fingerprint (2048 bits):
  - Radius 2, bit vector representation
```

### Cross-Validation Strategy
```
5-Fold Stratified K-Fold
â”œâ”€â”€ Training Set (80%): Model training
â””â”€â”€ Test Set (20%): Performance evaluation
   â”œâ”€â”€ ROC-AUC
   â”œâ”€â”€ Accuracy
   â”œâ”€â”€ Precision
   â””â”€â”€ Recall
```

### Class Imbalance Handling
```
XGBoost:       scale_pos_weight = (n_negative / n_positive)
Random Forest: class_weight="balanced"
```

## Risk Classification

```
Prediction Probability | Risk Level | Interpretation
        < 0.30        |    Low     | Accept for screening
      0.30 - 0.70     |  Moderate  | Cautious evaluation
        > 0.70        |    High    | Recommend validation
```

## Applicability Domain

```
Max Tanimoto Similarity | Domain Status    | Confidence
        â‰¥ 0.60         | In-Domain        | High
      0.40 - 0.60      | Near-Boundary    | Moderate
        < 0.40         | Out-of-Domain    | Low (extrapolation)
```

## Scientific Disclaimer

**These predictions are for early-stage screening purposes only.** Models are trained on historical data and should not replace rigorous experimental validation. Always conduct:

- **In vitro** toxicity assays (hepatotoxicity, genotoxicity, hERG patch clamp)
- **Pharmacokinetic** studies (absorption, metabolism, clearance)
- **Animal studies** following regulatory guidelines (ICH, FDA)
- **Clinical trials** before therapeutic use

## References

1. Lipinski, C. A., et al. (1997). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." *Advanced Drug Delivery Reviews*, 23(1), 3-25.

2. Wildman, S. A., & Crippen, G. M. (1999). "Prediction of physicochemical parameters by higher order molecular field analysis." *Journal of Chemical Information and Modeling*, 39(5), 868-873.

3. Rogers, D., & Hahn, M. (2010). "Extended-connectivity fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742-754.

4. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 785-794.

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests welcome! Please:
1. Follow PEP 8 style guidelines
2. Add type hints and docstrings
3. Include unit tests
4. Update README if adding features

## Authors

[Your Name] - Initial development

## Citation

If you use this project in research, please cite:

```bibtex
@software{admet_profiler_2024,
  author = {Your Name},
  title = {In-Silico Multi-Endpoint ADMET Risk Profiler},
  year = {2024},
  url = {https://github.com/yourusername/in-silico-admet}
}
```

