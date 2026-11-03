"""
Testes unitários para o módulo `src.data_prep` (limpeza, auditoria de escala e derivadas).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data_prep import (
    DATA_DICTIONARY,
    TARGET_VARS,
    TEAM_COL,
    clean_data,
    load_dataset,
    load_raw_dataset,
)


class TestUnitDataPrep(unittest.TestCase):
    """Valida a preparação, higienização e enriquecimento dos dados."""

    def test_clean_data_fixes_scale_and_extracts_team_abbreviation(self) -> None:
        raw_sample = pd.DataFrame({
            "MATCHUP": [" BOS vs. MIA ", "LAL @ GSW", "LAL @ GSW"],
            "GAME_DATE": ["2024-01-10", "2024-01-11", "2024-01-11"],
            "MIN": [240.0, 240.0, 240.0],
            "PTS": [115, 110, 110],
            "REB": [45, 42, 42],
            "AST": [28, 25, 25],
            "STL": [8, 7, 7],
            "BLK": [5, 4, 4],
            "PF": [18, 20, 20],
            "FG3M": [15, 12, 12],
            "W_PCT": [650.0, 0.550, 0.550],    # 650.0 -> deve ser corrigido para 0.650
            "FG_PCT": [485.0, 0.470, 0.470],   # 485.0 -> deve ser corrigido para 0.485
            "FG3_PCT": [412.0, 0.360, 0.360],  # 412.0 -> deve ser corrigido para 0.412
            "FT_PCT": [825.0, 0.780, 0.780],   # 825.0 -> deve ser corrigido para 0.825
        })

        cleaned = clean_data(raw_sample)

        # Duplicados exatos devem ser removidos (3 linhas -> 2 linhas)
        self.assertEqual(len(cleaned), 2)

        # TEAM_ABBREVIATION extraída de MATCHUP
        self.assertIn(TEAM_COL, cleaned.columns)
        self.assertIn("BOS", cleaned[TEAM_COL].values)
        self.assertIn("LAL", cleaned[TEAM_COL].values)

        # Correções de escala (÷ 1000) nas colunas de proporção
        bos_row = cleaned[cleaned[TEAM_COL] == "BOS"].iloc[0]
        self.assertAlmostEqual(bos_row["W_PCT"], 0.650, places=3)
        self.assertAlmostEqual(bos_row["FG_PCT"], 0.485, places=3)
        self.assertAlmostEqual(bos_row["FG3_PCT"], 0.412, places=3)
        self.assertAlmostEqual(bos_row["FT_PCT"], 0.825, places=3)

    def test_clean_data_removes_negative_counts(self) -> None:
        df = pd.DataFrame({
            "TEAM_ABBREVIATION": ["BOS", "LAL"],
            "PTS": [110, -5],
            "REB": [44, 40],
        })
        cleaned = clean_data(df)
        self.assertTrue(np.isnan(cleaned.loc[1, "PTS"]))
        self.assertEqual(cleaned.loc[0, "PTS"], 110.0)

    def test_load_real_and_processed_datasets(self) -> None:
        raw_df = load_raw_dataset()
        clean_df = load_dataset()
        self.assertGreater(len(raw_df), 2000)
        self.assertEqual(len(clean_df), 2238)
        for col in TARGET_VARS + [TEAM_COL]:
            self.assertIn(col, clean_df.columns)
            self.assertIn(col, DATA_DICTIONARY)


if __name__ == "__main__":
    unittest.main()
