"""Feature extraction from SMILES strings."""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from config import (
    FINGERPRINT_BITS,
    FINGERPRINT_RADIUS,
    DESCRIPTOR_NAMES,
)
from utils import setup_logging

logger = setup_logging(__name__)


def is_valid_smiles(smiles: str) -> bool:
    """
    Validate SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(smiles, str) or pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def compute_descriptors(smiles: str) -> np.ndarray:
    """
    Compute 7 physicochemical descriptors.

    Args:
        smiles: SMILES string

    Returns:
        Array of 7 descriptor values

    Raises:
        ValueError: If SMILES invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
        ]
    )


def compute_fingerprint(smiles: str):
    """
    Compute 2048-bit Morgan fingerprint.

    Args:
        smiles: SMILES string

    Returns:
        Morgan fingerprint

    Raises:
        ValueError: If SMILES invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )


def featurize_smiles(smiles: str) -> np.ndarray:
    """
    Generate complete feature vector (2055 dimensions).

    Combines 7 descriptors + 2048-bit Morgan fingerprint.

    Args:
        smiles: SMILES string

    Returns:
        Feature vector (2055 dims)

    Raises:
        ValueError: If SMILES invalid
    """
    descriptors = compute_descriptors(smiles)
    fp = compute_fingerprint(smiles)
    fp_array = np.array(fp)
    return np.concatenate([descriptors, fp_array])


def featurize_batch(
    smiles_list: list, remove_invalid: bool = True
) -> Tuple[np.ndarray, list]:
    """
    Featurize multiple SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        remove_invalid: If True, skip invalid SMILES; else raise error

    Returns:
        Tuple of (X: feature matrix, valid_smiles: list)

    Raises:
        ValueError: If no valid SMILES found
    """
    features = []
    valid_smiles = []

    for smiles in smiles_list:
        if not is_valid_smiles(smiles):
            if remove_invalid:
                logger.debug(f"Skipping invalid SMILES: {smiles}")
                continue
            else:
                raise ValueError(f"Invalid SMILES: {smiles}")

        try:
            feat = featurize_smiles(smiles)
            features.append(feat)
            valid_smiles.append(smiles)
        except Exception as e:
            if remove_invalid:
                logger.debug(f"Error featurizing {smiles}: {e}")
                continue
            else:
                raise

    if len(features) == 0:
        raise ValueError("No valid SMILES could be featurized")

    X = np.array(features)
    logger.info(f"Featurized {len(valid_smiles)} molecules")
    return X, valid_smiles


def get_descriptor_names() -> list:
    """Get list of all feature names."""
    return DESCRIPTOR_NAMES
