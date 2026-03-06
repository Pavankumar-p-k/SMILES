import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
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

X = df.drop(columns=[col for col in LABEL_COLUMNS if col in df.columns], errors="ignore")

# Drop non-feature columns and fill missing values
if "smiles" in X.columns:
    X = X.drop(columns=["smiles"])
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

y = df[target]

print("Class distribution:")
print(y.value_counts())

# Calculate scale_pos_weight
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\nScale pos weight: {scale_pos_weight:.2f} (negatives={neg_count}, positives={pos_count})")

# Feature scaling (StandardScaler for continuous features)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Hyperparameter grid
param_grid = {
    'max_depth': [4, 6, 8],
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.85, 1.0],
}

# Base model with class weighting
base_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    eval_metric='logloss'
)

# GridSearchCV with 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("\nTuning hyperparameters...")
grid_search.fit(X_scaled, y)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# Train final model on full training set
best_model = grid_search.best_estimator_
best_model.fit(X_scaled, y)

# Cross-validation evaluation
aucs = []
prs = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_model.fit(X_train, y_train)
    probs = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)

    aucs.append(auc)
    prs.append(pr)
    print(f"Fold {fold+1} ROC-AUC: {auc:.4f}, PR-AUC: {pr:.4f}")

print(f"\nMean ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean PR-AUC:  {np.mean(prs):.4f} ± {np.std(prs):.4f}")

# Save model and scaler
os.makedirs("models_research", exist_ok=True)
joblib.dump(best_model, "models_research/SR_MMP_xgb.pkl")
joblib.dump(scaler, "models_research/scaler_xgb.pkl")

print("\nModel and scaler saved.")
