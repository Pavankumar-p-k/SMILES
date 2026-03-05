"""Unit tests for predictor.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from predictor import ADMETPredictor, classify_domain


class DummyProbModel:
    """Minimal predict_proba-compatible model for tests."""

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, X):
        x = np.asarray(X)
        n_rows = int(x.shape[0])
        return np.column_stack(
            [
                np.full(n_rows, 1.0 - self.probability, dtype=float),
                np.full(n_rows, self.probability, dtype=float),
            ]
        )


def _build_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


class PredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

        bundle = {
            "models": {
                "DILI": DummyProbModel(0.10),
                "BBB": DummyProbModel(0.20),
                "Ames": DummyProbModel(0.30),
                "hERG": DummyProbModel(0.40),
            },
            "metadata": {"created_utc": "test"},
        }

        self.model_path = self.tmpdir / "multitask_admet_model.pkl"
        self.fp_path = self.tmpdir / "training_fingerprints.pkl"

        joblib.dump(bundle, self.model_path)
        joblib.dump([_build_fp("CCO"), _build_fp("CCN")], self.fp_path)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_classify_domain(self) -> None:
        self.assertEqual(classify_domain(0.80), "In-Domain")
        self.assertEqual(classify_domain(0.55), "Near-Boundary")
        self.assertEqual(classify_domain(0.10), "Out-of-Domain")

    def test_predict_single_valid(self) -> None:
        predictor = ADMETPredictor(
            model_path=self.model_path,
            training_fp_path=self.fp_path,
            models_dir=self.tmpdir,
            enable_explainability=False,
        )
        result = predictor.predict_single("CCO", include_shap=False)

        self.assertTrue(result["valid"])
        self.assertEqual(set(result["predictions"].keys()), {"DILI", "BBB", "Ames", "hERG"})
        self.assertAlmostEqual(result["predictions"]["DILI"], 0.10, places=6)
        self.assertIn(result["ad_status"], {"In-Domain", "Near-Boundary", "Out-of-Domain"})
        self.assertIsNotNone(result["ad_similarity"])

    def test_predict_single_invalid_smiles(self) -> None:
        predictor = ADMETPredictor(
            model_path=self.model_path,
            training_fp_path=self.fp_path,
            models_dir=self.tmpdir,
            enable_explainability=False,
        )
        result = predictor.predict_single("not_a_smiles", include_shap=False)

        self.assertFalse(result["valid"])
        self.assertEqual(result["error"], "Invalid SMILES")

    def test_predict_batch(self) -> None:
        predictor = ADMETPredictor(
            model_path=self.model_path,
            training_fp_path=self.fp_path,
            models_dir=self.tmpdir,
            enable_explainability=False,
        )
        results = predictor.predict_batch(["CCO", "invalid"], include_shap=False)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["valid"])
        self.assertFalse(results[1]["valid"])


if __name__ == "__main__":
    unittest.main()
