# Master Review and Documentation: In-Silico ADMET

## 1. Project Overview

`in_silico_admet` is a machine-learning platform for early ADMET screening of small molecules from SMILES.

Primary goal:
- Predict risk probabilities for four endpoints:
  - `DILI` (Drug-Induced Liver Injury)
  - `BBB` (Blood-Brain Barrier penetration risk)
  - `Ames` mutagenicity
  - `hERG` cardiotoxicity risk

Why this is useful:
- It reduces wet-lab burden by triaging candidates early.
- It standardizes risk scoring across multiple toxicity/PK-relevant endpoints.
- It provides explainability (SHAP) and applicability-domain checks to improve decision confidence.

---

## 2. One-Command Execution

Use this single command from `in_silico_admet/`:

```powershell
.\run_all.ps1 -LaunchApp
```

What this does:
1. Verifies/generates required sample endpoint data if missing.
2. Trains multitask models (`train_multitask.py`).
3. Runs unit tests (`unittest discover`).
4. Runs full system verification (`test_system.py`).
5. Launches Streamlit UI (`app.py`).

Useful variants:
- No app launch:
  - `.\run_all.ps1`
- Only launch UI after previous setup:
  - `.\run_all.ps1 -SkipTrain -SkipTests -SkipSystemCheck -LaunchApp`

---

## 3. Technology Stack

Core stack used in code:
- Python
- RDKit (chemistry parsing, fingerprints, structure rendering)
- NumPy / Pandas (numerics + tabular data)
- scikit-learn (calibration, metrics, CV, classical ML tools)
- XGBoost (tree boosting models)
- SHAP + Matplotlib (explainability plots)
- Streamlit (web UI)
- ReportLab (PDF report generation)
- Joblib (model artifact persistence)

Environment snapshot from local `venv`:
- Python `3.11.7`
- NumPy `2.3.5`
- Pandas `2.3.3`
- scikit-learn `1.7.2`
- XGBoost `3.2.0`
- RDKit `2025.09.4`
- Streamlit `1.54.0`
- ReportLab `4.4.10`
- SHAP `0.50.0`

---

## 4. Data and Features

### 4.1 Dataset

Primary multitask training data:
- `data/admet_multitask.csv`

Current observed stats:
- Raw rows: `30`
- Valid SMILES: `30`
- Unique canonical SMILES: `29` (one duplicate removed during training)

Endpoint prevalence in raw file:
- `DILI`: 21 positive / 9 negative (70.0% positive)
- `BBB`: 19 positive / 11 negative (63.3% positive)
- `AMES`: 15 positive / 15 negative (50.0% positive)
- `hERG`: 21 positive / 9 negative (70.0% positive)

### 4.2 Feature Engineering

Feature vector size: `2055` per molecule.

Composition:
- 7 descriptors:
  - MolWt, LogP, H-bond donors, H-bond acceptors, rotatable bonds, TPSA, aromatic rings
- 2048-bit Morgan fingerprint:
  - Radius `2`
  - Bit length `2048`

Files:
- `featurizer.py`
- `config.py`

---

## 5. Model Architecture and Training

Main production training pipeline:
- `train_multitask.py`

Approach:
- One calibrated classifier per endpoint.
- Base estimator: XGBoost (`XGBClassifier`) via `_default_estimator_factory`.
- Probability calibration: `CalibratedClassifierCV` with sigmoid (`config.CALIBRATION_METHOD`).
- Data split strategy: scaffold-disjoint K-fold (Bemis-Murcko scaffold grouping).
- Handles endpoint missing labels using per-endpoint masks.
- Handles class imbalance using `scale_pos_weight`.

Artifacts produced:
- `models/multitask_admet_model.pkl`
- `models/multitask_admet_model.metrics.json`
- `models/training_fingerprints.pkl`

Backward compatibility:
- Predictor can also load endpoint-specific artifacts (`DILI_model.pkl`, etc.) if bundle is absent.

---

## 6. Inference, Risk Scoring, and AD Domain

Inference layer:
- `predictor.py`

Pipeline per molecule:
1. Validate SMILES.
2. Featurize to 2055-dim vector.
3. Predict endpoint probabilities using `predict_proba`.
4. Compute applicability-domain status via max Tanimoto similarity to training fingerprints.
5. Optionally compute SHAP values and top features.

Applicability Domain thresholds (`config.AD_THRESHOLDS`):
- `In-Domain` if similarity >= 0.7
- `Near-Boundary` if similarity >= 0.5
- `Out-of-Domain` otherwise

Risk aggregation:
- `risk_aggregator.py`
- Weighted composite score:
  - DILI: 0.3
  - hERG: 0.3
  - Ames: 0.2
  - BBB: 0.2
- Amplification: +5% per endpoint with probability >= 0.7
- Category thresholds:
  - Low: [0.00, 0.33)
  - Moderate: [0.33, 0.66)
  - High: [0.66, 1.00]

---

## 7. UI, Tables, Images, and Graphs

### 7.1 Streamlit UI (`app.py`)

Single-molecule mode:
- Input SMILES and optional SHAP.
- Renders:
  - Molecule image (RDKit)
  - Endpoint metric cards
  - Composite risk + recommendation text
  - AD status and similarity
  - SHAP waterfall plot
  - Top SHAP features table
  - Highlighted substructure image from positive fingerprint bits
- PDF report generation and download

Batch mode:
- Upload CSV with `smiles` column.
- Returns tabular output:
  - validity, AD status/similarity, endpoint probabilities, composite score, risk category
- Downloadable predictions CSV.

### 7.2 PDF Reporting (`report_generator.py`)

PDF includes:
- Molecular structure image
- Endpoint probability/risk table
- Composite row
- Applicability domain section
- Screening-use disclaimer

### 7.3 Additional Research Plots

Scripts:
- `evaluate_external_multitask.py`
- `shap_analysis.py`

Generated plot types (when those scripts are run):
- ROC curves (`plots/roc_curve_<endpoint>.png`)
- Precision-recall curves (`plots/pr_curve_<endpoint>.png`)
- Confusion matrices (`plots/confusion_matrix_<endpoint>.png`)
- SHAP summary bar plots (`plots/shap_summary_<endpoint>.png`)
- SHAP dependence plots (`plots/shap_dependence_<endpoint>_<n>.png`)

Current repository state:
- `plots/` and `results/` are not currently populated in this workspace.

---

## 8. Current Measured Model Values (From Saved Metrics)

From `models/multitask_admet_model.metrics.json`:

Dataset/training metadata:
- created_utc: `2026-02-13T14:00:22.040917+00:00`
- n_molecules: `29`
- feature_dim: `2055`
- endpoints: `DILI, BBB, Ames, hERG`
- n_splits: `3`
- calibration_folds: `3`
- random_state: `42`

Cross-validation summary (mean +/- std):
- `DILI`: ROC-AUC `0.725 +/- 0.154`, PR-AUC `0.808 +/- 0.200`, MCC `0.274 +/- 0.289`
- `BBB`: ROC-AUC `0.568 +/- 0.201`, PR-AUC `0.689 +/- 0.286`, MCC `0.183 +/- 0.259`
- `Ames`: ROC-AUC `0.756 +/- 0.175`, PR-AUC `0.829 +/- 0.142`, MCC `0.285 +/- 0.206`
- `hERG`: ROC-AUC `0.725 +/- 0.154`, PR-AUC `0.808 +/- 0.200`, MCC `0.274 +/- 0.289`

Interpretation:
- Strongest ranking signal appears on `Ames` in this small dataset.
- `BBB` is weakest and likely data-limited.
- Variance is notable due small sample size and scaffold split constraints.

---

## 9. Validation Status (Latest Local Checks)

Unit tests:
- `11` tests run
- `11` passed
- `1` skipped (SHAP backend compatibility guard)

System verification:
- Import checks: pass
- Version checks: pass
- Directory checks: pass
- Data checks: pass
- Model artifact checks: pass
- Core functionality checks: pass

---

## 10. Engineering Review

### Strengths
- Clear separation of modules: training, prediction, explainability, UI, reporting.
- Good robustness in predictor loading (bundle + backward compatibility).
- Practical AD handling and structured risk aggregation.
- Unit tests cover core predictor/training/explainability paths.

### Risks / Limitations
- Dataset is very small; metrics can be unstable across folds.
- Some legacy research scripts use different feature conventions (2056 with extra descriptors).
- `requirements.txt` pins differ from current active `venv` versions, creating reproducibility ambiguity.
- SHAP behavior can vary by model/backend versions (already partially guarded in tests).

### Recommended Next Improvements
1. Lock and sync environment via exact reproducible file (`requirements-lock.txt` or `pip-tools`/`uv` lock).
2. Add CI pipeline to run `run_all.py` (without UI launch) on every commit.
3. Expand dataset and add external validation summaries to `results/`.
4. Add calibration/reliability plots and threshold-optimized decision policy per endpoint.
5. Standardize legacy scripts on the same feature schema as `train_multitask.py`.

