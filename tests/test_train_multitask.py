"""Unit tests for train_multitask.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import train_multitask
from train_multitask import load_multitask_dataset, scaffold_kfold_indices, train_multitask_model


def _sample_dataframe() -> pd.DataFrame:
    smiles = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "CCF",
        "CCS",
        "COC",
        "COCC",
        "CC(=O)O",
        "CC(=O)N",
        "CC(C)O",
        "CC(C)N",
        "CC(C)C",
        "CCCO",
        "CCCN",
        "c1ccccc1",
        "c1ccncc1",
        "c1ccccc1O",
        "c1ccccc1N",
        "C1CCCCC1",
        "C1=CC=CN=C1",
        "CCOC",
        "CCNC",
    ]

    rows = []
    for idx, smi in enumerate(smiles):
        rows.append(
            {
                "smiles": smi,
                "dili": int(("N" in smi) or (idx % 3 == 0)),
                "bbb": int(("c1" in smi) or ("C1" in smi) or (idx % 4 == 0)),
                "ames": int(("N" in smi) or ("Br" in smi) or ("Cl" in smi)),
                "herg": int(("Cl" in smi) or ("Br" in smi) or (("c1" in smi) and (idx % 2 == 0))),
            }
        )

    return pd.DataFrame(rows)


class TrainMultitaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_scaffold_kfold_indices_disjoint(self) -> None:
        smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "c1ccncc1", "CCCl"]
        splits = scaffold_kfold_indices(smiles, n_splits=3, random_state=11)

        self.assertEqual(len(splits), 3)

        all_test = set()
        for train_idx, test_idx in splits:
            self.assertTrue(set(train_idx).isdisjoint(set(test_idx)))
            all_test.update(test_idx.tolist())

        self.assertEqual(all_test, set(range(len(smiles))))

    def test_load_multitask_dataset_filters_invalid_and_duplicates(self) -> None:
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCO", "invalid", "CCN"],
                "dili": [0, 0, 1, 1],
                "bbb": [1, 1, 0, 0],
                "ames": [0, 0, 1, 1],
                "herg": [0, 0, 1, 1],
            }
        )
        csv_path = self.tmpdir / "toy.csv"
        df.to_csv(csv_path, index=False)

        dataset = load_multitask_dataset(csv_path)
        self.assertEqual(len(dataset.smiles), 2)
        self.assertEqual(dataset.X.shape[0], 2)
        self.assertEqual(dataset.Y.shape[1], 4)

    def test_train_multitask_model_with_custom_estimator_factory(self) -> None:
        df = _sample_dataframe()
        csv_path = self.tmpdir / "train.csv"
        output_path = self.tmpdir / "bundle.pkl"
        df.to_csv(csv_path, index=False)

        def estimator_factory(_scale_pos_weight: float, seed: int):
            return LogisticRegression(max_iter=300, solver="liblinear", random_state=seed)

        old_models_dir = train_multitask.MODELS_DIR
        train_multitask.MODELS_DIR = self.tmpdir

        try:
            bundle = train_multitask_model(
                dataset_csv=csv_path,
                output_model_path=output_path,
                n_splits=2,
                calibration_folds=2,
                random_state=7,
                estimator_factory=estimator_factory,
                save_metrics_json=False,
            )
        finally:
            train_multitask.MODELS_DIR = old_models_dir

        self.assertTrue(output_path.exists())
        self.assertIn("models", bundle)
        self.assertGreaterEqual(len(bundle["models"]), 2)
        self.assertTrue((self.tmpdir / "training_fingerprints.pkl").exists())


if __name__ == "__main__":
    unittest.main()
