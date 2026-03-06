import sys

try:
    import deepchem as dc
except ModuleNotFoundError:
    print("Module 'deepchem' not found. Install it before running this script.")
    print("Recommended (conda): conda install -c conda-forge deepchem")
    print(
        "Or try pip (may require extra index): pip install deepchem --extra-index-url https://pypi.anaconda.org/deepchem/simple"
    )
    sys.exit(1)

import pandas as pd
import os
import numpy as np

print("Downloading Tox21 dataset...")

tasks, datasets, transformers = dc.molnet.load_tox21()

train_dataset, valid_dataset, test_dataset = datasets

print("Train samples:", len(train_dataset))
print("Valid samples:", len(valid_dataset))
print("Test samples:", len(test_dataset))


# Convert DeepChem dataset to pandas
def dc_to_df(dataset):
    # Support various dataset.X formats (sparse, 3D arrays, lists)
    X = dataset.X
    try:
        arr = np.asarray(X)
    except Exception:
        arr = np.array(list(X))

    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    df = pd.DataFrame(arr)
    df["smiles"] = list(dataset.ids)
    for i, task in enumerate(tasks):
        df[task] = dataset.y[:, i]
    return df


train_df = dc_to_df(train_dataset)

os.makedirs("data_large", exist_ok=True)
train_df.to_csv("data_large/tox21_train.csv", index=False)

valid_df = dc_to_df(valid_dataset)
valid_df.to_csv("data_large/tox21_valid.csv", index=False)

test_df = dc_to_df(test_dataset)
test_df.to_csv("data_large/tox21_test.csv", index=False)

print("Saved to data_large/tox21_train.csv")
print("Saved to data_large/tox21_valid.csv")
print("Saved to data_large/tox21_test.csv")
