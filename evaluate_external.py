import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os

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

# Load model
model_path = "models_research/SR_MMP_rf.pkl"
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

model = joblib.load(model_path)

# Load external test set
test_path = "data_large/tox21_test.csv"
if not os.path.exists(test_path):
    print(f"Test data not found: {test_path}, using validation set instead.")
    test_path = "data_large/tox21_valid.csv"
    if not os.path.exists(test_path):
        print(f"Validation data not found either: {test_path}")
        sys.exit(1)

df_test = pd.read_csv(test_path)

target = "SR-MMP"
df_test = df_test.dropna(subset=[target])

X_test = df_test.drop(
    columns=[col for col in LABEL_COLUMNS if col in df_test.columns],
    errors="ignore",
)

# Drop non-feature columns and fill missing values
if "smiles" in X_test.columns:
    X_test = X_test.drop(columns=["smiles"])
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

y_test = df_test[target]

# Predict
probs = model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, probs)
pr = average_precision_score(y_test, probs)

print(f"External ROC-AUC: {roc:.4f}")
print(f"External PR-AUC : {pr:.4f}")
