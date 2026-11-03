"""
zscore.py
=========
Deteção de outliers pelo Z-Score Clássico (desvio padrão da média) e Z-Score Modificado (MAD).
"""

from __future__ import annotations

import pandas as pd

ZSCORE_THEORY: dict[str, str] = {
    "key": "zscore",
    "code": "Z-Score",
    "name": "Z-Score Clássico (Desvio Padrão) & Modificado (MAD)",
    "category": "Univariado · Paramétrico / Robusto",
    "formula": "z_i = (x_i - x̄) / s   (|z_i| > 3,0)   |   Mz_i = 0,6745·|x_i - Md| / MAD   (> 3,5)",
    "theory": (
        "Quantifica a distância de cada observação x_i à média amostral x̄ em unidades de desvio padrão s. "
        "Pela propriedade da distribuição Normal N(μ, σ²), 99,73% das observações situam-se em [x̄ ± 3s]; "
        "logo |z_i| > 3,0 assinala observações raras (0,27% nas caudas). A variante de Iglewicz & Hoaglin "
        "substitui média e desvio padrão pela mediana e pelo Desvio Absoluto Mediano (MAD)."
    ),
    "why_use": (
        "Padroniza variáveis de diferentes unidades numa escala adimensional comum. Quando a distribuição "
        "é muito assimétrica, o Z-Score modificado (MAD) evita que valores extremos inflamem o desvio padrão."
    ),
}


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Máscara booleana pelo Z-score clássico |z| > threshold."""
    num = pd.to_numeric(series, errors="coerce")
    std = num.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=series.index)
    z = (num - num.mean()) / std
    return (z.abs() > threshold).fillna(False)


def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Z-score modificado (Iglewicz & Hoaglin) — robusto, baseado na MAD."""
    num = pd.to_numeric(series, errors="coerce")
    median = num.median()
    mad = (num - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=series.index)
    mz = 0.6745 * (num - median) / mad
    return (mz.abs() > threshold).fillna(False)

