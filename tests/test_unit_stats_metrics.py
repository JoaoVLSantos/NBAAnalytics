"""
Testes unitários para o módulo `src.stats_metrics` (estatística descritiva univariada).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.stats_metrics import calculate_univariate_stats, interpret_cv, interpret_skewness


class TestUnitStatsMetrics(unittest.TestCase):
    """Valida os cálculos de tendência central, dispersão, separatrizes e forma."""

    def test_calculate_univariate_stats_standard_series(self) -> None:
        s = pd.Series([10, 20, 20, 30, 40, np.nan])
        res = calculate_univariate_stats(s)

        self.assertEqual(res["n"], 5)
        self.assertEqual(res["n_missing"], 1)
        self.assertEqual(res["mean"], 24.0)
        self.assertEqual(res["median"], 20.0)
        self.assertEqual(res["mode"], 20.0)
        self.assertEqual(res["min"], 10.0)
        self.assertEqual(res["max"], 40.0)
        self.assertEqual(res["range"], 30.0)
        self.assertEqual(res["q1"], 20.0)
        self.assertEqual(res["q3"], 30.0)
        self.assertEqual(res["iqr"], 10.0)
        self.assertAlmostEqual(res["variance"], 130.0, places=2)
        self.assertAlmostEqual(res["std"], 11.40, places=2)
        self.assertIsNotNone(res["skewness"])

    def test_calculate_univariate_stats_empty_and_single_value(self) -> None:
        empty = pd.Series([np.nan, np.nan])
        res_empty = calculate_univariate_stats(empty)
        self.assertEqual(res_empty["n"], 0)
        self.assertEqual(res_empty["n_missing"], 2)
        self.assertIsNone(res_empty["mean"])
        self.assertIsNone(res_empty["std"])

        single = pd.Series([42.0])
        res_single = calculate_univariate_stats(single)
        self.assertEqual(res_single["n"], 1)
        self.assertEqual(res_single["mean"], 42.0)
        self.assertIsNone(res_single["variance"])
        self.assertIsNone(res_single["skewness"])

    def test_interpretations_skewness_and_cv(self) -> None:
        self.assertIn("Indeterminada", interpret_skewness(None))
        self.assertEqual(interpret_skewness(0.1), "Aproximadamente simétrica")
        self.assertIn("Moderadamente", interpret_skewness(0.7))
        self.assertIn("à direita", interpret_skewness(0.7))
        self.assertIn("Fortemente", interpret_skewness(-1.4))
        self.assertIn("à esquerda", interpret_skewness(-1.4))

        self.assertEqual(interpret_cv(None), "Indeterminada")
        self.assertEqual(interpret_cv(10.0), "Baixa dispersão")
        self.assertEqual(interpret_cv(22.0), "Dispersão moderada")
        self.assertEqual(interpret_cv(35.0), "Alta dispersão")


if __name__ == "__main__":
    unittest.main()

