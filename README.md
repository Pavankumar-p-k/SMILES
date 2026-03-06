# In-Silico ADMET Risk Platform

A production-oriented cheminformatics and machine learning platform for early-stage ADMET risk screening.

This repository includes:
- Multi-endpoint toxicity prediction models
- Applicability-domain scoring
- Two dashboards (Streamlit + Holographic React UI)
- Report generation and training workflows

## What This Project Does

The platform predicts risk probabilities for four core endpoints from a SMILES input:
- `BBB` (Blood-Brain Barrier risk)
- `DILI` (Drug-Induced Liver Injury)
- `Ames` (Mutagenicity risk)
- `hERG` (Cardiac ion-channel risk)

It also provides:
- Composite risk category (`Low`, `Moderate`, `High`)
- Applicability-domain status using Tanimoto similarity
- Organ mapping in UI:
  - Brain -> BBB
  - Liver -> DILI
  - Heart -> hERG
  - Genetic/DNA -> Ames

## Core Features

- Multi-endpoint prediction pipeline (`predictor.py`)
- Risk aggregation (`risk_aggregator.py`)
- Applicability-domain classification (`applicability_domain.py`)
- Lipinski checks (`lipinski.py`)
- Explainability helpers (`explainability.py`, optional SHAP path)
- PDF-style report generation (`report_generator.py`)
- Training and evaluation scripts for baseline and multitask workflows

## Dashboards

### 1) Streamlit Dashboard
- File: `app.py`
- Purpose: analytical dashboard for single/batch workflows and reporting
- Default local URL: `http://127.0.0.1:8501`

### 2) Holographic React Dashboard
- Folder: `holographic-brain-ui/`
- Purpose: interactive anatomical visualization with live ADMET overlay
- Uses local API bridge: `holographic-brain-ui/tools/admet_api_server.py`
- Default local URL: `http://127.0.0.1:5173`

## One-Click Local Start (Both Dashboards)

From repo root:

```bat
run-holographic-ui.bat
```

This launcher:
- Starts Streamlit dashboard
- Starts ADMET API bridge backend
- Starts Holographic React UI dev server

Optional args:

```bat
run-holographic-ui.bat <react_host> <react_port> <streamlit_port>
```

Example:

```bat
run-holographic-ui.bat 127.0.0.1 5173 8501
```

## Local Setup

### Requirements
- Python 3.10+
- Node.js 18+
- npm

### Python install

```bash
pip install -r requirements.txt
```

### Frontend install

```bash
cd holographic-brain-ui
npm install
```

## Prediction API (Local Bridge)

Base URL (default): `http://127.0.0.1:5051`

Endpoints:
- `GET /health`
- `POST /predict`

Example payload:

```json
{
  "smiles": "CCO",
  "include_shap": false
}
```

## Model + Data Artifacts

- Models: `models/`
- Research models: `models_research/`
- Core datasets: `data/`
- Extended datasets: `data_large/`
- Tests: `tests/`

## Repository Layout

```text
in_silico_admet/
|- app.py
|- predictor.py
|- risk_aggregator.py
|- report_generator.py
|- train_models.py
|- train_multitask.py
|- data/
|- models/
|- tests/
|- ui/
|- holographic-brain-ui/
|  |- run-ui.bat
|  |- tools/admet_api_server.py
|  |- src/
|- run-holographic-ui.bat
```

## Risk Interpretation

Probability to level mapping:
- `< 0.30` -> `Low`
- `0.30 to <0.70` -> `Moderate`
- `>= 0.70` -> `High`

Composite category comes from weighted endpoint aggregation in `risk_aggregator.py`.

## Deployment (Recommended)

- Frontend: Vercel
- Backend/API: Render Web Service
- Streamlit: Render Web Service (second service)

Set frontend env variable:
- `VITE_ADMET_API_URL=https://<your-backend-service>.onrender.com`

## Scientific and Regulatory Note

This platform is intended for early-stage screening support only.
Predictions must be validated with experimental assays and standard preclinical workflows.

## License

MIT (or project-specific license as applicable).
