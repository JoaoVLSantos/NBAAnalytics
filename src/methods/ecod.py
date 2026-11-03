"""
ecod.py
=======
ECOD (Empirical Cumulative Distribution-based Outlier Detection — Li et al., 2022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ECOD_THEORY: dict[str, str] = {
    "key": "ecod",
    "code": "ECOD",
    "name": "Empirical Cumulative Distribution-based Outlier Detection (2022)",
    "category": "Multivariado / Univariado · Livre de Hiperparâmetros",
    "formula": "O(x_i) = max( -∑ log F̂_j(x_ij), -∑ log(1 - F̂_j(x_ij)), -∑ log p̂_skew(x_ij) )",
    "theory": (
        "Estima de forma não-paramétrica a Função de Distribuição Acumulada Empírica (ECDF) de cada variável "
        "e mede a probabilidade de cauda esquerda F̂_j(x) e direita 1 - F̂_j(x) de cada ponto em todas as dimensões, "
        "ponderando pela direção da assimetria (skewness) de cada coluna. Agrega as raridades marginais por "
        "log-probabilidade negativa para pontuar anomalias conjuntas."
    ),
    "why_use": (
        "Não requer ajuste de hiperparâmetros de distância ou vizinhança, é extremamente rápido O(n·d·log n), "
        "interpreta diretamente em qual cauda a anomalia ocorreu e lida naturalmente com variáveis assimétricas."
    ),
}


def detect_outliers_ecod(
    df: pd.DataFrame, cols: list[str], contamination: float = 0.05
) -> pd.Series:
    """Deteção de outliers via ECOD (usa PyOD localmente ou implementação vetorial equivalente em serverless)."""
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(data)
    if n < 3:
        return pd.Series(False, index=df.index)

    X = data.to_numpy(dtype=float)
    try:
        from pyod.models.ecod import ECOD

        model = ECOD(contamination=contamination)
        model.fit(X)
        is_outlier = model.labels_.astype(bool)
    except ImportError:
        # Implementação vetorial direta da equação de Li et al. (2022) (sem dependência de numba/pyod)
        u_l = np.zeros_like(X, dtype=float)
        u_r = np.zeros_like(X, dtype=float)
        u_s = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            col_vals = X[:, j]
            f_left = stats.rankdata(col_vals, method="max") / (n + 1.0)
            f_right = stats.rankdata(-col_vals, method="max") / (n + 1.0)
            u_l[:, j] = -np.log(np.clip(f_left, 1e-12, 1.0))
            u_r[:, j] = -np.log(np.clip(f_right, 1e-12, 1.0))
            skew_j = stats.skew(col_vals)
            u_s[:, j] = u_l[:, j] if skew_j < 0 else u_r[:, j]

        scores = np.maximum(u_l.sum(axis=1), np.maximum(u_r.sum(axis=1), u_s.sum(axis=1)))
        threshold = np.quantile(scores, 1.0 - contamination)
        is_outlier = scores > threshold

    labels = pd.Series(False, index=df.index)
    labels.loc[data.index] = is_outlier
    return labels
