"""
stats_metrics.py
================
Medidas de estatística descritiva univariada.

Função principal:
    calculate_univariate_stats(series) -> dict

Convenções adotadas:
    * Variância e desvio padrão são AMOSTRAIS (ddof=1), i.e. divisor n-1.
    * Quartis calculados por interpolação linear (padrão do pandas,
      equivalente ao "type 7" de Hyndman & Fan / R).
    * Assimetria = coeficiente de Fisher-Pearson AJUSTADO (G1),
      obtido com scipy.stats.skew(..., bias=False).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Casas decimais utilizadas no arredondamento
_DEC = 2          # medidas na unidade da variável
_DEC_SHAPE = 3    # medidas adimensionais de forma (assimetria)


def _round(value: Any, ndigits: int = _DEC) -> float | None:
    """Converte para float nativo e arredonda; NaN/inf passam a None.

    Devolver tipos nativos (e não numpy.float64) garante que o dicionário
    é diretamente serializável em JSON e "limpo" para o template Jinja2.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, ndigits)


def _empty_result(n_missing: int) -> dict:
    """Resultado padrão quando não existem observações válidas."""
    keys = [
        "mean", "median", "mode", "min", "max", "range", "q1", "q3",
        "iqr", "variance", "std", "cv_pct", "skewness",
    ]
    result = {k: None for k in keys}
    result.update({"n": 0, "n_missing": n_missing, "n_modes": 0})
    return result


def calculate_univariate_stats(series: pd.Series) -> dict:
    """Calcula as principais medidas descritivas de uma variável numérica.

    Parameters
    ----------
    series : pd.Series
        Coluna numérica (valores não numéricos são convertidos em NaN
        e, tal como os valores em falta, ignorados no cálculo).

    Returns
    -------
    dict
        n, n_missing                -> contagens
        mean, median, mode, n_modes -> tendência central
        min, max, range, q1, q3, iqr-> posição
        variance, std, cv_pct       -> dispersão (amostral, ddof=1)
        skewness                    -> forma (Fisher-Pearson ajustado)
    """
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    n = int(clean.size)
    n_missing = int(numeric.size - n)

    if n == 0:
        return _empty_result(n_missing)

    # ---------------- Tendência central ----------------
    mean = clean.mean()
    median = clean.median()
    modes = clean.mode()                       # pode haver mais do que uma moda
    mode = modes.iloc[0] if not modes.empty else np.nan

    # ---------------- Posição ----------------
    minimum, maximum = clean.min(), clean.max()
    amplitude = maximum - minimum
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1

    # ---------------- Dispersão (amostral) ----------------
    # Com n = 1 a variância amostral é indefinida (divisão por zero).
    variance = clean.var(ddof=1) if n > 1 else np.nan
    std = clean.std(ddof=1) if n > 1 else np.nan

    # CV% = (s / média) * 100 — indefinido se a média for 0.
    cv_pct = (std / mean) * 100 if (mean != 0 and not np.isnan(std)) else np.nan

    # ---------------- Forma ----------------
    # O estimador ajustado (bias=False) requer n >= 3. Para séries
    # constantes o scipy devolve NaN e emite um RuntimeWarning: silenciamos.
    if n >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            skewness = stats.skew(clean.to_numpy(dtype=float), bias=False)
    else:
        skewness = np.nan

    return {
        "n": n,
        "n_missing": n_missing,
        "mean": _round(mean),
        "median": _round(median),
        "mode": _round(mode),
        "n_modes": int(modes.size),
        "min": _round(minimum),
        "max": _round(maximum),
        "range": _round(amplitude),
        "q1": _round(q1),
        "q3": _round(q3),
        "iqr": _round(iqr),
        "variance": _round(variance),
        "std": _round(std),
        "cv_pct": _round(cv_pct),
        "skewness": _round(skewness, _DEC_SHAPE),
    }


def interpret_skewness(skewness: float | None) -> str:
    """Interpretação qualitativa do coeficiente de assimetria.

    Regra prática (Bulmer, 1979):
        |g| < 0.5        -> aproximadamente simétrica
        0.5 <= |g| < 1   -> moderadamente assimétrica
        |g| >= 1         -> fortemente assimétrica
    """
    if skewness is None:
        return "Indeterminada (dados insuficientes)"
    magnitude = abs(skewness)
    if magnitude < 0.5:
        return "Aproximadamente simétrica"
    side = "à direita (positiva)" if skewness > 0 else "à esquerda (negativa)"
    level = "Moderadamente" if magnitude < 1 else "Fortemente"
    return f"{level} assimétrica {side}"


def interpret_cv(cv_pct: float | None) -> str:
    """Classificação habitual da dispersão relativa pelo CV%."""
    if cv_pct is None:
        return "Indeterminada"
    cv = abs(cv_pct)
    if cv < 15:
        return "Baixa dispersão"
    if cv < 30:
        return "Dispersão moderada"
    return "Alta dispersão"
