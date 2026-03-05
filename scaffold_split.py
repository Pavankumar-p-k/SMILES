"""
Scaffold-based train/test split using Bemis-Murcko scaffolds.
Prevents chemical scaffold leakage.
"""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random


def generate_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(smiles_list, test_size=0.2, random_state=42):
    scaffold_to_indices = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        scaffold = generate_scaffold(smi)
        scaffold_to_indices[scaffold].append(i)

    scaffolds = list(scaffold_to_indices.keys())
    random.seed(random_state)
    random.shuffle(scaffolds)

    train_idx, test_idx = [], []

    for scaffold in scaffolds:
        if len(test_idx) / len(smiles_list) < test_size:
            test_idx.extend(scaffold_to_indices[scaffold])
        else:
            train_idx.extend(scaffold_to_indices[scaffold])

    return train_idx, test_idx
