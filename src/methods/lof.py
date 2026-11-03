"""
lof.py
======
LOF (Local Outlier Factor — Breunig et al., 2000).
"""

from __future__ import annotations

import warnings
import pandas as pd

LOF_THEORY: dict[str, str] = {
    "key": "lof",
    "code": "LOF",
    "name": "Local Outlier Factor (Breunig et al., 2000)",
    "category": "Multivariado · Densidade Local Relativa",
    "formula": "LOF_k(x_i) = [ (1/k) ∑_{o ∈ N_k(x_i)} lrd_k(o) ] / lrd_k(x_i)   (Outlier se LOF_k ≫ 1)",
    "theory": (
        "Calcula a densidade local de alcançabilidade (local reachability density — lrd_k) de cada observação "
        "e compara-a com a densidade média dos seus k vizinhos mais próximos. Um valor LOF ≈ 1 indica que o ponto "
        "tem densidade idêntica à dos seus vizinhos; um LOF significativamente maior que 1 indica que o ponto está "
        "numa região muito menos densa do que a sua vizinhança local."
    ),
    "why_use": (
        "Deteta 'outliers locais' que métodos globais (como K-NN ou DBSCAN com ε único) falham em identificar "
        "quando diferentes grupos (ex.: equipas de ritmos distintos) possuem densidades internas diferentes."
    ),
}


def detect_outliers_lof(
    df: pd.DataFrame, cols: list[str], contamination: float = 0.05, n_neighbors: int = 20
) -> pd.Series:
    """Deteção de anomalias por densidade local relativa via LOF (Local Outlier Factor)."""
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    k = min(n_neighbors, max(2, len(data) - 1))
    if len(data) < 4:
        return pd.Series(False, index=df.index)

    X = StandardScaler().fit_transform(data.to_numpy(dtype=float))
    lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        preds = lof.fit_predict(X)  # -1 = outlier, 1 = inlier

    labels = pd.Series(False, index=df.index)
    labels.loc[data.index] = preds == -1
    return labels

