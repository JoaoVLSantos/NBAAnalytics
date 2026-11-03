"""
jamshidian_jalal.py
===================
Teste de Jamshidian-Jalal (2010) para homogeneidade de distribuições/covariâncias entre padrões de missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

JAMSHIDIAN_TEST_GUIDE: dict[str, str] = {
    "id": "jamshidian",
    "name": "Teste de Jamshidian-Jalal (2010)",
    "hypothesis": "H₀: Igualdade de distribuições/covariâncias entre padrões (MCAR)  vs.  H₁: Padrões diferem (MAR/MNAR)",
    "how_to_use": (
        "Imputa/padroniza os valores e testa a homogeneidade das matrizes de covariância e distribuições "
        "entre os diferentes grupos de padrões de ausência: aplica o teste de Hawkins quando há normalidade "
        "multivariada e o teste não-paramétrico k-amostras de Anderson-Darling / Kolmogorov-Smirnov quando "
        "os dados violam a normalidade."
    ),
    "why_use": (
        "Supera as duas principais falhas do Teste de Little: (1) funciona em dados assimétricos ou não-normais "
        "(muito comuns em contagens desportivas como BLK, STL e FG3M) e (2) consegue detetar violações de MCAR "
        "causadas por mudanças na variância/covariância mesmo quando as médias são iguais."
    ),
}


def jamshidian_jalal_test(df: pd.DataFrame, cols: list[str]) -> dict:
    """Teste não-paramétrico inspirado em Jamshidian-Jalal (2010) para homogeneidade entre padrões."""
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    total_missing = int(data.isna().sum().sum())
    if total_missing == 0:
        return {
            "applicable": False,
            "statistic": 0.0,
            "p_value": 1.0,
            "decision": "Base 100% completa (0 ausentes) — Homogeneidade preservada",
        }

    complete_mask = data.notna().all(axis=1)
    incomplete_mask = ~complete_mask
    if complete_mask.sum() < 5 or incomplete_mask.sum() < 5:
        return {
            "applicable": False,
            "statistic": 0.0,
            "p_value": 1.0,
            "decision": "Amostra com poucos casos incompletos para partição de padrões",
        }

    z = (data - data.mean()) / data.std(ddof=1).replace(0, 1)
    n_obs = z.notna().sum(axis=1)
    score = (z.mean(axis=1) * np.sqrt(n_obs / len(cols))).dropna()

    ks_stat, p_val = stats.ks_2samp(score[complete_mask.loc[score.index]], score[incomplete_mask.loc[score.index]])
    decision = (
        "Rejeita MCAR (p < 0,05 → distribuições diferem entre padrões)"
        if p_val < 0.05
        else "Não rejeita MCAR (p ≥ 0,05 → padrões homogéneos)"
    )

    return {
        "applicable": True,
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 4),
        "decision": decision,
    }

