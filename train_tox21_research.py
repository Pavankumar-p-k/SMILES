import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os
import sys

# Tox21 endpoint label columns
LABEL_COLUMNS = [
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
]

# Load dataset
data_path = "data_large/tox21_train.csv"
if not os.path.exists(data_path):
    print(f"Data file not found: {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)

# Select one endpoint
target = "SR-MMP"

df = df.dropna(subset=[target])

X = df.drop(
    columns=[col for col in LABEL_COLUMNS if col in df.columns],
    errors="ignore",
)

# Drop non-feature columns and fill missing values
if "smiles" in X.columns:
    X = X.drop(columns=["smiles"])
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

y = df[target]

print("Class distribution:")
print(y.value_counts())

# Model
model = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

aucs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    aucs.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# Save model
os.makedirs("models_research", exist_ok=True)
joblib.dump(model, "models_research/SR_MMP_rf.pkl")

print("Model saved.")
