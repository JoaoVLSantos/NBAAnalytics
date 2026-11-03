"""
dbscan.py
=========
DBSCAN (Density-Based Spatial Clustering of Applications with Noise — Ester et al., 1996).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DBSCAN_THEORY: dict[str, str] = {
    "key": "dbscan",
    "code": "DBSCAN",
    "name": "Density-Based Spatial Clustering of Applications with Noise (1996)",
    "category": "Multivariado · Agrupamento por Densidade",
    "formula": "N_ε(x_i) = {x_j : d(x_i, x_j) ≤ ε}   →   Ruído/Outlier (label = -1) se |N_ε| < min_samples",
    "theory": (
        "Define clusters como regiões contínuas de alta densidade separadas por regiões de baixa densidade. "
        "Um ponto é central (core point) se tiver pelo menos 'min_samples' pontos dentro de um raio ε (eps). "
        "Pontos que não são centrais nem estão no raio ε de nenhum ponto central não pertencem a nenhum cluster "
        "e são rotulados diretamente como ruído/anomalia (label = -1)."
    ),
    "why_use": (
        "Não força uma percentagem fixa de contaminação, descobre agrupamentos de formato arbitrário (não apenas "
        "esféricos) e isola naturalmente observações em regiões esparsas como ruído."
    ),
}


def detect_outliers_dbscan(
    df: pd.DataFrame, cols: list[str], eps: float | None = None, min_samples: int = 8
) -> pd.Series:
    """Deteção de anomalias via DBSCAN (pontos de ruído com label == -1)."""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < min_samples:
        return pd.Series(False, index=df.index)

    X = StandardScaler().fit_transform(data.to_numpy(dtype=float))
    d = X.shape[1]
    effective_eps = eps if eps is not None else (0.65 if d == 1 else round(0.85 * np.sqrt(d), 2))

    clustering = DBSCAN(eps=effective_eps, min_samples=min_samples).fit(X)
    labels = pd.Series(False, index=df.index)
    labels.loc[data.index] = clustering.labels_ == -1
    return labels

