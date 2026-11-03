"""
Testes unitários para `src.outlier_detection` e todos os métodos em `src.methods`.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data_prep import DATA_DICTIONARY
from src.outlier_detection import (
    OUTLIER_THEORY_GUIDE,
    detect_outliers_dbscan,
    detect_outliers_ecod,
    detect_outliers_grubbs,
    detect_outliers_iqr,
    detect_outliers_knn,
    detect_outliers_lof,
    detect_outliers_modified_zscore,
    detect_outliers_pyod,
    detect_outliers_zscore,
    grubbs_test_summary,
    iqr_bounds,
    summarize_columns_inconsistencies_and_outliers,
)


class TestUnitOutlierDetection(unittest.TestCase):
    """Valida cada método de deteção de outliers univariado e multivariado."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        normal_pts = rng.normal(110, 5, size=98)
        # Adiciona 2 outliers extremos claros
        pts = np.concatenate([normal_pts, [185.0, 40.0]])
        reb = np.concatenate([rng.normal(44, 3, size=98), [80.0, 15.0]])
        ast = np.concatenate([rng.normal(25, 3, size=98), [55.0, 5.0]])
        self.df = pd.DataFrame({
            "TEAM_ABBREVIATION": ["BOS"] * 50 + ["LAL"] * 50,
            "PTS": pts,
            "REB": reb,
            "AST": ast,
            "FG_PCT": [0.48] * 99 + [0.52],
        })

    def test_iqr_and_zscore_methods(self) -> None:
        s = self.df["PTS"]
        low, high = iqr_bounds(s, k=1.5)
        self.assertLess(low, high)

        mask_iqr_mod = detect_outliers_iqr(s, k=1.5)
        mask_iqr_ext = detect_outliers_iqr(s, k=3.0)
        self.assertTrue(mask_iqr_mod.iloc[98])
        self.assertTrue(mask_iqr_mod.iloc[99])
        self.assertGreaterEqual(mask_iqr_mod.sum(), mask_iqr_ext.sum())

        mask_z = detect_outliers_zscore(s, threshold=3.0)
        mask_mod_z = detect_outliers_modified_zscore(s, threshold=3.5)
        self.assertTrue(mask_z.iloc[98])
        self.assertTrue(mask_mod_z.iloc[98])

    def test_grubbs_method(self) -> None:
        s = self.df["PTS"]
        summary = grubbs_test_summary(s, alpha=0.05)
        self.assertTrue(summary["is_outlier"])
        self.assertGreater(summary["g_stat"], summary["g_crit"])

        mask_grubbs = detect_outliers_grubbs(s, alpha=0.05)
        self.assertGreaterEqual(mask_grubbs.sum(), 2)
        self.assertTrue(mask_grubbs.iloc[98])

    def test_ecod_knn_dbscan_lof_methods(self) -> None:
        cols = ["PTS", "REB", "AST"]

        mask_ecod = detect_outliers_ecod(self.df, cols, contamination=0.05)
        mask_knn = detect_outliers_knn(self.df, cols, contamination=0.05)
        mask_dbscan = detect_outliers_dbscan(self.df, cols)
        mask_lof = detect_outliers_lof(self.df, cols, contamination=0.05)

        for mask in (mask_ecod, mask_knn, mask_dbscan, mask_lof):
            self.assertEqual(len(mask), len(self.df))
            self.assertTrue(mask.iloc[98])
            self.assertTrue(mask.iloc[99])

        # Dispatcher unificado
        self.assertTrue(detect_outliers_pyod(self.df, cols, method="ecod").equals(mask_ecod))
        self.assertTrue(detect_outliers_pyod(self.df, cols, method="knn").equals(mask_knn))
        self.assertTrue(detect_outliers_pyod(self.df, cols, method="dbscan").equals(mask_dbscan))
        self.assertTrue(detect_outliers_pyod(self.df, cols, method="lof").equals(mask_lof))

    def test_column_summary_inconsistencies_and_outliers(self) -> None:
        raw_df = self.df.copy()
        raw_df.loc[0, "FG_PCT"] = 480.0  # Inconsistência de escala ×1000 no bruto
        rows = summarize_columns_inconsistencies_and_outliers(raw_df, self.df, DATA_DICTIONARY)

        by_col = {r["column"]: r for r in rows}
        self.assertEqual(by_col["FG_PCT"]["inconsistent"], 1)
        self.assertFalse(by_col["TEAM_ABBREVIATION"]["is_quantitative"])
        self.assertTrue(by_col["PTS"]["is_quantitative"])
        self.assertGreaterEqual(by_col["PTS"]["iqr_mod"], 2)
        self.assertEqual(len(OUTLIER_THEORY_GUIDE), 7)


if __name__ == "__main__":
    unittest.main()

