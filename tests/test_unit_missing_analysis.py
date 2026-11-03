"""
Testes unitários para o módulo `src.missing_analysis` e os 4 testes em `src.methods`.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.missing_analysis import (
    MISSING_MECHANISMS,
    MISSING_TESTS_GUIDE,
    evaluate_missing_tests,
    impute,
    missing_by_group,
    missing_summary,
)


class TestUnitMissingAnalysis(unittest.TestCase):
    """Valida contagens de missing, imputações e os 4 testes estatísticos."""

    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        n = 150
        self.df = pd.DataFrame({
            "TEAM_ABBREVIATION": rng.choice(["BOS", "LAL", "GSW"], size=n),
            "PTS": rng.normal(112, 12, size=n),
            "REB": rng.normal(44, 6, size=n),
            "AST": rng.normal(26, 5, size=n),
            "STL": rng.normal(7.5, 2.5, size=n),
        })

    def test_missing_summary_and_by_group(self) -> None:
        df = self.df.copy()
        df.loc[0:9, "PTS"] = np.nan
        summary = missing_summary(df)

        self.assertEqual(summary.loc["PTS", "missing"], 10)
        self.assertGreater(summary.loc["PTS", "pct_missing"], 0.0)
        self.assertEqual(summary.loc["REB", "missing"], 0)

        by_group = missing_by_group(df, "TEAM_ABBREVIATION", ["PTS", "REB"])
        self.assertIn("PTS", by_group.columns)
        self.assertEqual(len(by_group), 3)

    def test_impute_strategies(self) -> None:
        df = pd.DataFrame({"PTS": [100.0, 110.0, 120.0, np.nan]})

        imp_median = impute(df, ["PTS"], strategy="median")
        self.assertEqual(imp_median["PTS"].isna().sum(), 0)
        self.assertEqual(imp_median.loc[3, "PTS"], 110.0)

        imp_mean = impute(df, ["PTS"], strategy="mean")
        self.assertEqual(imp_mean.loc[3, "PTS"], 110.0)

        imp_zero = impute(df, ["PTS"], strategy="zero")
        self.assertEqual(imp_zero.loc[3, "PTS"], 0.0)

        with self.assertRaises(ValueError):
            impute(df, ["PTS"], strategy="invalid")

    def test_four_missing_tests_complete_and_simulated(self) -> None:
        cols = ["PTS", "REB", "AST", "STL"]

        # 1. Base completa (0 missing)
        res_real = evaluate_missing_tests(self.df, cols, target_col="PTS", sim_mode="real")
        self.assertEqual(res_real["little"]["p_value"], 1.0)
        self.assertEqual(res_real["jamshidian"]["p_value"], 1.0)
        self.assertEqual(res_real["listing_schlittgen"]["p_value"], 1.0)
        self.assertEqual(res_real["fairclough"]["p_value"], 1.0)

        # 2. Cenário MCAR simulado
        res_mcar = evaluate_missing_tests(self.df, cols, target_col="PTS", sim_mode="mcar")
        self.assertTrue(res_mcar["little"]["applicable"])
        self.assertGreater(res_mcar["little"]["df"], 0)
        self.assertIsNotNone(res_mcar["little"]["p_value"])
        self.assertIsNotNone(res_mcar["jamshidian"]["p_value"])
        self.assertIsNotNone(res_mcar["listing_schlittgen"]["p_value"])
        self.assertIsNotNone(res_mcar["fairclough"]["p_value"])

        # 3. Cenário MAR simulado (ausência fortemente dependente de REB + AST)
        res_mar = evaluate_missing_tests(self.df, cols, target_col="PTS", sim_mode="mar")
        self.assertLess(res_mar["listing_schlittgen"]["p_value"], 0.05)
        self.assertLess(res_mar["fairclough"]["p_value"], 0.05)

    def test_mechanisms_and_guides_metadata(self) -> None:
        self.assertEqual(len(MISSING_MECHANISMS), 3)
        self.assertEqual([m["code"] for m in MISSING_MECHANISMS], ["MCAR", "MAR", "MNAR"])
        self.assertEqual(len(MISSING_TESTS_GUIDE), 4)


if __name__ == "__main__":
    unittest.main()
