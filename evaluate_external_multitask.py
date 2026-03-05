import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, confusion_matrix
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# RDKit feature extraction (same as training)
def extract_rdkit_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp)
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'AromaticProportion': Descriptors.NumAromaticRings(mol) / max(1, Descriptors.RingCount(mol)),
        }
        return np.concatenate([fp_array, list(descriptors.values())])
    except Exception:
        return None

# Load test data
test_path = "data_large/tox21_test.csv"
if not os.path.exists(test_path):
    print(f"Test data not found: {test_path}, using validation set.")
    test_path = "data_large/tox21_valid.csv"

if not os.path.exists(test_path):
    print("No test data available.")
    sys.exit(1)

df_test = pd.read_csv(test_path)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Extract features
print("Extracting test features...")
X_list = []
for idx, row in df_test.iterrows():
    features = extract_rdkit_features(row['smiles'])
    if features is not None:
        X_list.append(features)
    else:
        X_list.append(np.zeros(2056))

X_test = np.array(X_list)

# Load preprocessing artifacts
var_threshold_path = "models/var_threshold.pkl"
scaler_path = "models/scaler.pkl"
if os.path.exists(var_threshold_path) and os.path.exists(scaler_path):
    var_threshold = joblib.load(var_threshold_path)
    scaler = joblib.load(scaler_path)
    X_test = var_threshold.transform(X_test)
    X_test_scaled = X_test.copy()
    X_test_scaled[:, -8:] = scaler.transform(X_test[:, -8:])
else:
    print("Preprocessing artifacts not found; evaluating on untransformed features.")
    X_test_scaled = X_test

# Define endpoints
endpoints = [
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
    'NR-ER-LBD', 'NR-PPAR-gamma'
]

# Evaluate external
external_results = []

for endpoint in endpoints:
    model_path = f"models/{endpoint}_xgb.pkl"
    if not os.path.exists(model_path):
        continue
    
    model = joblib.load(model_path)
    
    # Labels
    y_test = df_test[endpoint].values
    valid_idx = ~np.isnan(y_test)
    X_ep = X_test_scaled[valid_idx]
    y_ep = y_test[valid_idx].astype(int)
    
    if len(np.unique(y_ep)) < 2:
        continue
    
    # Predictions
    probs = model.predict_proba(X_ep)[:, 1]
    preds = model.predict(X_ep)
    
    roc_auc = roc_auc_score(y_ep, probs)
    pr_auc = average_precision_score(y_ep, probs)
    f1 = f1_score(y_ep, preds)
    ba = balanced_accuracy_score(y_ep, preds)
    
    result = {
        'Endpoint': endpoint,
        'Test_ROC_AUC': f"{roc_auc:.4f}",
        'Test_PR_AUC': f"{pr_auc:.4f}",
        'Test_F1': f"{f1:.4f}",
        'Test_BA': f"{ba:.4f}",
        'Test_N': len(y_ep),
    }
    external_results.append(result)
    
    print(f"{endpoint}: ROC={roc_auc:.4f}, PR={pr_auc:.4f}, F1={f1:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_ep, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.4f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve - {endpoint}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"plots/roc_curve_{endpoint}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot PR curve
    precision, recall, _ = precision_recall_curve(y_ep, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR (AUC={pr_auc:.4f})', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {endpoint}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"plots/pr_curve_{endpoint}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_ep, preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title(f'Confusion Matrix - {endpoint}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red', fontsize=12)
    plt.savefig(f"plots/confusion_matrix_{endpoint}.png", dpi=150, bbox_inches='tight')
    plt.close()

# Save results
external_df = pd.DataFrame(external_results)
external_df.to_csv("results/external_evaluation.csv", index=False)
print("\n\nExternal Evaluation Summary:")
print(external_df)
