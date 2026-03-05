import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def extract_rdkit_features(smiles):
    """Generate Morgan fingerprint + descriptors for a single SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp)
        descriptors = {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol),
            "AromaticProportion": Descriptors.NumAromaticRings(mol)
            / max(1, Descriptors.RingCount(mol)),
        }
        return np.concatenate([fp_array, list(descriptors.values())])
    except Exception:
        return None


data_path = "data_large/tox21_train.csv"
df = pd.read_csv(data_path)
os.makedirs("plots", exist_ok=True)

print("Extracting training features for SHAP...")
X_list = []
for idx, row in df.iterrows():
    if idx % 500 == 0:
        print(f"  {idx}/{len(df)}")
    features = extract_rdkit_features(row["smiles"])
    if features is not None:
        X_list.append(features)
    else:
        X_list.append(np.zeros(2056))

X = np.array(X_list)

var_threshold_path = "models/var_threshold.pkl"
scaler_path = "models/scaler.pkl"
if os.path.exists(var_threshold_path) and os.path.exists(scaler_path):
    var_threshold = joblib.load(var_threshold_path)
    scaler = joblib.load(scaler_path)
    X = var_threshold.transform(X)
    X_scaled = X.copy()
    X_scaled[:, -8:] = scaler.transform(X[:, -8:])
else:
    print("Preprocessing artifacts not found; running SHAP on untransformed features.")
    X_scaled = X

top_endpoints = ["SR-MMP", "SR-p53", "NR-ER"]

for endpoint in top_endpoints:
    print(f"\nSHAP analysis for {endpoint}...")

    model_path = f"models/{endpoint}_xgb.pkl"
    if not os.path.exists(model_path):
        print(f"  Model not found for {endpoint}")
        continue

    model = joblib.load(model_path)

    y = df[endpoint].values
    valid_idx = ~np.isnan(y)
    X_ep = X_scaled[valid_idx]

    sample_idx = np.random.choice(len(X_ep), min(100, len(X_ep)), replace=False)
    X_sample = X_ep[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Summary - {endpoint}")
    plt.tight_layout()
    plt.savefig(f"plots/shap_summary_{endpoint}.png", dpi=150, bbox_inches="tight")
    plt.close()

    feature_importance = np.abs(shap_vals).mean(axis=0)
    top_3_features = np.argsort(feature_importance)[-3:][::-1]

    for i, feat_idx in enumerate(top_3_features):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat_idx, shap_vals, X_sample, show=False)
        plt.title(f"SHAP Dependence - {endpoint} (Feature {feat_idx})")
        plt.tight_layout()
        plt.savefig(
            f"plots/shap_dependence_{endpoint}_{i + 1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

print("\nSHAP analysis complete.")
