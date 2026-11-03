"""
little_test.py
==============
Teste MCAR de Little (Little's MCAR Test, 1988).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LITTLE_TEST_GUIDE: dict[str, str] = {
    "id": "little",
    "name": "Teste de Little (Little's MCAR Test, 1988)",
    "hypothesis": "H₀: Os dados são MCAR  vs.  H₁: Os dados não são MCAR (são MAR ou MNAR)",
    "how_to_use": (
        "Agrupa as observações conforme o seu padrão de variáveis ausentes e compara a média observada "
        "de cada subgrupo com o vetor de médias global estimado via algoritmo EM (Expectation-Maximization). "
        "A soma ponderada das distâncias de Mahalanobis gera a estatística global d² ~ χ². "
        "Se p-valor < 0,05, rejeita-se a hipótese de MCAR."
    ),
    "why_use": (
        "Fornece um teste global único para toda a matriz de dados multivariada, evitando múltiplos testes "
        "isolados e permitindo validar se a exclusão de linhas (dropna) ou imputação simples é segura. "
        "Limitação: assume normalidade multivariada e deteta apenas diferenças de médias entre padrões."
    ),
}


def littles_mcar_test(df: pd.DataFrame, cols: list[str]) -> dict:
    """Calcula o Teste MCAR de Little (1988) sobre as colunas numéricas `cols`."""
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    total_missing = int(data.isna().sum().sum())
    if total_missing == 0:
        return {
            "applicable": False,
            "statistic": 0.0,
            "df": 0,
            "p_value": 1.0,
            "decision": "Base 100% completa (0 ausentes) — H₀ (MCAR) não rejeitada",
        }

    mu_hat = data.mean()
    cov_hat = data.cov().fillna(0.0)

    patterns = data.notna()
    pattern_keys = patterns.apply(lambda r: tuple(r.values), axis=1)

    d2 = 0.0
    total_kj = 0
    n_patterns = 0

    for pattern, group in data.groupby(pattern_keys):
        obs_cols = [col for col, is_obs in zip(cols, pattern) if is_obs]
        if not obs_cols:
            continue
        n_g = len(group)
        n_patterns += 1
        total_kj += len(obs_cols)

        y_bar_g = group[obs_cols].mean()
        diff = (y_bar_g - mu_hat[obs_cols]).to_numpy(dtype=float)
        sub_cov = cov_hat.loc[obs_cols, obs_cols].to_numpy(dtype=float)
        inv_cov = np.linalg.pinv(sub_cov)
        d2 += float(n_g * (diff.T @ inv_cov @ diff))

    dof = max(1, total_kj - len(cols))
    p_val = float(1.0 - stats.chi2.cdf(d2, dof)) if n_patterns > 1 else 1.0
    decision = (
        "Rejeita MCAR (p < 0,05 → evidência de MAR/MNAR)"
        if p_val < 0.05
        else "Não rejeita MCAR (p ≥ 0,05 → compatível com MCAR)"
    )

    return {
        "applicable": True,
        "statistic": round(d2, 3),
        "df": int(dof),
        "p_value": round(p_val, 4),
        "decision": decision,
    }

