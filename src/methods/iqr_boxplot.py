"""
iqr_boxplot.py
==============
Deteção de outliers pelo Intervalo Interquartil (Cercas de Tukey / Boxplot, 1977).
"""

from __future__ import annotations

import pandas as pd

IQR_THEORY: dict[str, str] = {
    "key": "iqr",
    "code": "IQR / Boxplot",
    "name": "Intervalo Interquartil (Cercas de Tukey) & Extremos",
    "category": "Univariado · Não-paramétrico",
    "formula": "Moderado: [Q₁ - 1,5·IQR, Q₃ + 1,5·IQR]  |  Extremo: [Q₁ - 3,0·IQR, Q₃ + 3,0·IQR]",
    "theory": (
        "Baseado nos quartis amostrais Q₁ (percentil 25) e Q₃ (percentil 75), onde IQR = Q₃ - Q₁ "
        "representa a dispersão dos 50% centrais dos dados. No Boxplot de Tukey (1977), os bigodes "
        "estendem-se até 1,5·IQR: valores fora da cerca interna (k = 1,5) são outliers moderados "
        "e valores fora da cerca externa (k = 3,0) são outliers severos/extremos."
    ),
    "why_use": (
        "Não assume distribuição normal e é resistente ao efeito de mascaramento (masking), pois Q₁ e Q₃ "
        "têm ponto de rutura (breakdown point) de 25%, não sendo distorcidos pelos próprios extremos."
    ),
}


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Cercas de Tukey: [Q1 - k·IQR, Q3 + k·IQR] (k=1.5 moderado, k=3.0 extremo)."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0, 0.0
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Máscara booleana dos outliers pelo critério IQR (NaN -> False)."""
    num = pd.to_numeric(series, errors="coerce")
    clean = num.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)
    lower, upper = iqr_bounds(clean, k)
    return ((num < lower) | (num > upper)).fillna(False)

