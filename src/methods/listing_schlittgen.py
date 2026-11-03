"""
listing_schlittgen.py
=====================
Teste de Listing e Schlittgen (LS, 1998) para comparação de variáveis auxiliares entre R_j = 1 e R_j = 0.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

LISTING_SCHLITTGEN_GUIDE: dict[str, str] = {
    "id": "listing_schlittgen",
    "name": "Teste de Listing e Schlittgen (LS, 1998)",
    "hypothesis": "H₀: E(Y_k | R_j = 1) = E(Y_k | R_j = 0) (MCAR)  vs.  H₁: A distribuição de Y_k difere quando Y_j falta (MAR)",
    "how_to_use": (
        "Para uma variável Y_j que possui valores ausentes, cria-se o indicador binário R_j (1 = presente, 0 = ausente) "
        "e divide-se cada variável auxiliar observada Y_k em dois grupos. Aplica-se o teste t de Student/Welch (paramétrico) "
        "ou o teste de Mann-Whitney-Wilcoxon (não-paramétrico) para comparar Y_k entre R_j = 1 e R_j = 0."
    ),
    "why_use": (
        "Ao contrário do teste global de Little, o teste LS identifica diretamente QUAL variável observada Y_k "
        "está associada à ausência de Y_j, sendo altamente intuitivo e orientando a escolha das variáveis "
        "preditoras para a imputação."
    ),
}


def listing_schlittgen_test(df: pd.DataFrame, target_col: str, aux_cols: list[str]) -> dict:
    """Teste de Listing e Schlittgen (1998): compara variáveis auxiliares entre R_j=1 e R_j=0."""
    is_missing = df[target_col].isna()
    n_mis = int(is_missing.sum())
    n_obs = int((~is_missing).sum())

    if n_mis < 2 or n_obs < 2:
        return {
            "applicable": False,
            "target_col": target_col,
            "n_missing": n_mis,
            "statistic": 0.0,
            "p_value": 1.0,
            "most_associated_col": "—",
            "decision": f"Coluna {target_col} sem ausências suficientes (n_missing = {n_mis})",
        }

    best_stat = 0.0
    min_p = 1.0
    best_col = aux_cols[0] if aux_cols else "—"

    for col in aux_cols:
        if col == target_col or col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        g_obs = s[~is_missing].dropna()
        g_mis = s[is_missing].dropna()
        if len(g_obs) < 2 or len(g_mis) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(g_obs, g_mis, equal_var=False)
        if abs(t_stat) > abs(best_stat):
            best_stat = float(t_stat)
            min_p = float(p_val)
            best_col = col

    n_tested = max(1, len([c for c in aux_cols if c != target_col and c in df.columns]))
    adj_p = min(1.0, min_p * n_tested)
    decision = (
        f"Rejeita MCAR (p = {adj_p:.4f} < 0,05; associado a {best_col} → MAR)"
        if adj_p < 0.05
        else f"Não rejeita MCAR (p = {adj_p:.4f} ≥ 0,05)"
    )

    return {
        "applicable": True,
        "target_col": target_col,
        "n_missing": n_mis,
        "statistic": round(best_stat, 3),
        "p_value": round(adj_p, 4),
        "most_associated_col": best_col,
        "decision": decision,
    }

