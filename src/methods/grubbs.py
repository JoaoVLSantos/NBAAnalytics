"""
grubbs.py
=========
Teste de Grubbs (Extreme Studentized Deviate — ESD, 1950) para deteção formal de outliers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

GRUBBS_THEORY: dict[str, str] = {
    "key": "grubbs",
    "code": "Grubbs (ESD)",
    "name": "Teste de Grubbs (Extreme Studentized Deviate, 1950)",
    "category": "Univariado · Teste de Hipótese Formal",
    "formula": "G = max |x_i - x̄| / s   vs.   G_crít = ((n-1)/√n) · √[ t²_(α/2n, n-2) / (n - 2 + t²_(α/2n, n-2)) ]",
    "theory": (
        "Teste estatístico formal para testar H₀: 'não existem outliers na amostra' contra H₁: 'o valor "
        "mais extremo é um outlier', assumindo normalidade aproximada. Calcula o maior desvio estudentizado "
        "G e compara-o com o valor crítico derivado da distribuição t de Student com n-2 graus de liberdade "
        "e correção de Bonferroni α/(2n). Aplicado de forma iterativa (ESD sequencial), remove o extremo e "
        "repete o teste até que G ≤ G_crít."
    ),
    "why_use": (
        "Diferente de limiares fixos (como |z| > 3), o Teste de Grubbs ajusta automaticamente o limiar crítico "
        "ao tamanho da amostra (n) e controla formalmente a taxa de falsos positivos (nível de significância α = 5%)."
    ),
}


def grubbs_test_summary(series: pd.Series, alpha: float = 0.05) -> dict:
    """Calcula a estatística G do Teste de Grubbs (1950), o valor crítico G_crít e o p-valor."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n < 3:
        return {"g_stat": 0.0, "g_crit": 0.0, "p_value": 1.0, "is_outlier": False}

    std = float(clean.std(ddof=1))
    if std == 0:
        return {"g_stat": 0.0, "g_crit": 0.0, "p_value": 1.0, "is_outlier": False}

    mean = float(clean.mean())
    deviations = (clean - mean).abs()
    g_stat = float(deviations.max() / std)

    t_crit = float(stats.t.ppf(1.0 - alpha / (2.0 * n), df=n - 2))
    g_crit = float(((n - 1) / np.sqrt(n)) * np.sqrt((t_crit ** 2) / (n - 2 + t_crit ** 2)))

    ratio_sq = (g_stat * np.sqrt(n) / (n - 1)) ** 2
    if ratio_sq >= 1.0:
        p_val = 0.0
    else:
        t_obs = np.sqrt((n - 2) * ratio_sq / (1.0 - ratio_sq))
        p_val = float(min(1.0, 2.0 * n * (1.0 - stats.t.cdf(t_obs, df=n - 2))))

    return {
        "g_stat": round(g_stat, 3),
        "g_crit": round(g_crit, 3),
        "p_value": round(p_val, 4),
        "is_outlier": bool(g_stat > g_crit),
    }


def detect_outliers_grubbs(series: pd.Series, alpha: float = 0.05, max_iter: int = 150) -> pd.Series:
    """Teste de Grubbs iterativo (Sequential ESD): identifica todos os pontos que excedem G_crít."""
    work = pd.to_numeric(series, errors="coerce").dropna().copy()
    outlier_indices: list = []

    for _ in range(max_iter):
        n = len(work)
        if n < 3:
            break
        std = float(work.std(ddof=1))
        if std == 0:
            break
        mean = float(work.mean())
        devs = (work - mean).abs()
        max_idx = devs.idxmax()
        g_stat = float(devs.loc[max_idx] / std)

        t_crit = float(stats.t.ppf(1.0 - alpha / (2.0 * n), df=n - 2))
        g_crit = float(((n - 1) / np.sqrt(n)) * np.sqrt((t_crit ** 2) / (n - 2 + t_crit ** 2)))

        if g_stat > g_crit:
            outlier_indices.append(max_idx)
            work = work.drop(index=max_idx)
        else:
            break

    mask = pd.Series(False, index=series.index)
    if outlier_indices:
        mask.loc[outlier_indices] = True
    return mask

