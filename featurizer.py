"""
featurizer.py

Responsible for converting SMILES strings into numerical features
for ADMET and toxicity prediction.

Outputs:
- Physicochemical descriptors (interpretable)
- Morgan fingerprints (structural information)

Author: Your Name
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import pandas as pd


# -----------------------------
# Descriptor configuration
# -----------------------------

DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "RotatableBonds": Descriptors.NumRotatableBonds,
    "AromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}


# -----------------------------
# Core featurization functions
# -----------------------------


def smiles_to_mol(smiles: str):
    """
    Convert SMILES to RDKit Mol object.

    Returns None if invalid.
    """
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def compute_descriptors(mol):
    """
    Compute physicochemical descriptors from RDKit Mol.

    Returns:
        dict of descriptor_name: value
    """
    return {name: func(mol) for name, func in DESCRIPTOR_FUNCTIONS.items()}


def compute_morgan_fingerprint(mol, radius=2, n_bits=512):
    """
    Compute Morgan fingerprint as numpy array.

    Returns:
        np.ndarray of shape (n_bits,)
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=int)


def featurize_smiles(smiles: str):
    """
    Full featurization pipeline for a single SMILES.

    Returns:
        descriptor_vector (dict)
        fingerprint_vector (np.ndarray)

    Raises:
        ValueError if SMILES is invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    descriptors = compute_descriptors(mol)
    fingerprint = compute_morgan_fingerprint(mol)

    return descriptors, fingerprint


# -----------------------------
# Batch featurization
# -----------------------------


def featurize_dataframe(df: pd.DataFrame, smiles_col: str = "smiles"):
    """
    Featurize a DataFrame containing SMILES.

    Returns:
        X_desc (pd.DataFrame)
        X_fp (np.ndarray)
        valid_indices (list)
    """
    descriptor_list = []
    fingerprint_list = []
    valid_indices = []

    for idx, smiles in enumerate(df[smiles_col]):
        try:
            desc, fp = featurize_smiles(smiles)
            descriptor_list.append(desc)
            fingerprint_list.append(fp)
            valid_indices.append(idx)
        except ValueError:
            # Skip invalid SMILES
            continue

    X_desc = pd.DataFrame(descriptor_list)
    X_fp = np.vstack(fingerprint_list)

    return X_desc, X_fp, valid_indices
