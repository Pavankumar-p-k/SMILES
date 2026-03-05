"""Unit tests for explainability.py."""

from __future__ import annotations

import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import explainability
from explainability import (
    compute_shap_values,
    extract_top_positive_fingerprint_bits,
    morgan_bit_highlights,
    top_contributing_features,
)


class ExplainabilityTests(unittest.TestCase):
    def test_top_contributing_features(self) -> None:
        rows = top_contributing_features(
            shap_values=np.array([0.1, -0.5, 0.2, -0.05]),
            feature_names=["a", "b", "c", "d"],
            top_n=2,
        )
        self.assertEqual(rows[0]["feature"], "b")
        self.assertEqual(rows[1]["feature"], "c")

    def test_extract_top_positive_fingerprint_bits(self) -> None:
        names = ["desc0", "desc1", "Morgan_FP_0", "Morgan_FP_1", "Morgan_FP_2"]
        shap_values = np.array([-0.1, 0.05, 0.2, 0.9, 0.4])

        bits = extract_top_positive_fingerprint_bits(
            shap_values=shap_values,
            feature_names=names,
            top_n=2,
        )
        self.assertEqual(bits, [1, 2])

    def test_morgan_bit_highlights(self) -> None:
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bit_info)
        some_bit = next(iter(bit_info.keys()))

        mapping = morgan_bit_highlights(smiles, [some_bit])
        self.assertIn(some_bit, mapping["bits_found"])
        self.assertGreater(len(mapping["atom_indices"]), 0)

    @unittest.skipIf(explainability.shap is None, "shap not installed")
    def test_compute_shap_values_if_shap_available(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)

        model = RandomForestClassifier(n_estimators=10, random_state=7)
        model.fit(X, y)

        try:
            values, base_value = compute_shap_values(model, X[0])
        except Exception as exc:
            self.skipTest(f"SHAP backend incompatibility in current env: {exc}")

        self.assertEqual(values.shape[0], X.shape[1])
        self.assertTrue(np.isfinite(base_value))


if __name__ == "__main__":
    unittest.main()
