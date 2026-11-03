"""
knn.py
======
K-NN (K-Nearest Neighbors) para Deteção de Anomalias (Ramaswamy et al., 2000).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KNN_THEORY: dict[str, str] = {
    "key": "knn",
    "code": "K-NN Anomalias",
    "name": "K-Nearest Neighbors para Deteção de Anomalias (2000)",
    "category": "Multivariado · Baseado em Distância",
    "formula": "Score_KNN(x_i) = d(x_i, N_k(x_i))   (distância euclidiana ao k-ésimo vizinho mais próximo)",
    "theory": (
        "Após padronizar as variáveis (StandardScaler), projeta cada partida num espaço métrico d-dimensional "
        "e calcula a distância euclidiana de cada observação x_i aos seus k vizinhos mais próximos. "
        "Observações situadas em regiões normais possuem vizinhos muito próximos (distância pequena), enquanto "
        "anomalias multivariadas ficam isoladas e apresentam grande distância ao k-ésimo vizinho."
    ),
    "why_use": (
        "Captura combinações atípicas entre múltiplas estatísticas (ex.: pontuação alta com pouquíssimos minutos "
        "ou combinação incomum de REB, AST e TOV) que passariam despercebidas olhando uma coluna de cada vez."
    ),
}


def detect_outliers_knn(
    df: pd.DataFrame, cols: list[str], contamination: float = 0.05, n_neighbors: int = 5
) -> pd.Series:
    """Deteção de anomalias via K-Nearest Neighbors (PyOD ou scikit-learn NearestNeighbors)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) <= n_neighbors:
        return pd.Series(False, index=df.index)

    X = StandardScaler().fit_transform(data.to_numpy(dtype=float))
    try:
        from pyod.models.knn import KNN

        model = KNN(contamination=contamination, n_neighbors=n_neighbors)
        model.fit(X)
        is_outlier = model.labels_.astype(bool)
    except ImportError:
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
        distances, _ = nn.kneighbors(X)
        kth_dist = distances[:, -1]
        threshold = np.quantile(kth_dist, 1.0 - contamination)
        is_outlier = kth_dist > threshold

    labels = pd.Series(False, index=df.index)
    labels.loc[data.index] = is_outlier
    return labels
