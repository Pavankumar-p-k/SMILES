"""Generate sample ADMET training datasets for testing."""

import pandas as pd
from pathlib import Path
import numpy as np

# Sample SMILES with realistic toxicity patterns
SAMPLE_SMILES = [
    # Low toxicity
    "CCO",
    "CC(C)C",
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "c1ccccc1",  # Benzene
    "CC(C)CC(=O)O",  # Isobutyric acid
    "CCCC",  # Butane
    "CC(C)O",  # Isopropanol
    "CC(C)(C)O",  # tert-Butanol
    "c1ccc(O)cc1",  # Phenol
    "CC(=O)N",  # Acetamide
    # Moderate toxicity
    "CC(C)Cl",  # Isopropyl chloride
    "CC(=O)Cl",  # Acetyl chloride
    "Cc1ccccc1",  # Toluene
    "CCc1ccccc1",  # Ethylbenzene
    "CC(C)(C)N",  # tert-Butylamine
    "CCCC(=O)O",  # Butyric acid
    "CC(C)CC(=O)N",  # Isovaleramic
    "c1ccc(Cl)cc1",  # Chlorobenzene
    "Cc1ccccc1O",  # o-Cresol
    "c1ccc(C)cc1",  # Toluene
    # High toxicity
    "CCN(CC)CC",  # Triethylamine
    "CC(C)N(C)C",  # Dimethylisopropylamine
    "c1ccc(cc1)N",  # Aniline
    "Cc1ccc(N)cc1",  # o-Toluidine
    "CC(=O)NC(=O)C",  # Acetamidobenzoic
    "CCN(CC)C(=O)C",  # N,N-Diethylacetamide
    "Cc1ccccc1N",  # o-Methylaniline
    "c1ccc(N(C)C)cc1",  # N,N-Dimethylaniline
    "CCN(C)C(=O)C",  # N,N-Diethylacetamide
    "c1ccc(S(=O)(=O)N)cc1",  # Benzenesulfonamide
]

# Realistic label assignment (simplified logic)
DILI_LABELS = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,  # Low tox
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,  # Moderate
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]  # High tox

BBB_LABELS = [
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    1,
]

AMES_LABELS = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]

HERG_LABELS = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


def generate_datasets():
    """Generate all sample datasets."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    datasets = {
        "dili": DILI_LABELS,
        "bbb": BBB_LABELS,
        "ames": AMES_LABELS,
        "herg": HERG_LABELS,
    }

    for name, labels in datasets.items():
        df = pd.DataFrame(
            {
                "smiles": SAMPLE_SMILES,
                "label": labels,
            }
        )

        csv_path = data_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Generated {csv_path} ({len(df)} molecules)")


if __name__ == "__main__":
    generate_datasets()
    print("\n✓ All sample datasets generated in data/")
