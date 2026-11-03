"""
fairclough_logistic.py
======================
Regressão Logística de Fairclough (1998 / 2002) para modelação da probabilidade de ausência P(M_j = 1 | X).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

FAIRCLOUGH_LOGISTIC_GUIDE: dict[str, str] = {
    "id": "fairclough",
    "name": "Regressão Logística de Fairclough (1998 / 2002)",
    "hypothesis": "H₀: β₁ = β₂ = ... = β_p = 0 (MCAR)  vs.  H₁: Pelo menos um preditor explica a probabilidade de ausência (MAR)",
    "how_to_use": (
        "Define-se a variável resposta binária M_j = 1 (se Y_j está ausente) e M_j = 0 (se observado) e ajusta-se "
        "um modelo de Regressão Logística: logit(P(M_j = 1 | X)) = β₀ + β₁X₁ + ... + β_pX_p usando as variáveis "
        "observadas X como explicativas. Avalia-se a significância global pelo Teste da Razão de Verossimilhança (χ²) "
        "e de cada variável pelo teste de Wald."
    ),
    "why_use": (
        "Permite testar simultaneamente múltiplas variáveis contínuas e categóricas (ex.: MIN, FGA, WL, Time) "
        "controlando o efeito umas das outras, identificar as covariáveis que governam o mecanismo MAR e "
        "construir pesos de propensão (Inverse Probability Weighting) ou análises de sensibilidade para MNAR."
    ),
}


def fairclough_logistic_test(df: pd.DataFrame, target_col: str, predictor_cols: list[str]) -> dict:
    """Regressão Logística de Fairclough: modela P(R_j = 0 | X) com Teste de Razão de Verossimilhança."""
    y = df[target_col].isna().astype(int)
    n_mis = int(y.sum())

    if n_mis < 3 or n_mis > len(df) - 3:
        return {
            "applicable": False,
            "target_col": target_col,
            "statistic": 0.0,
            "df": 0,
            "p_value": 1.0,
            "top_predictor": "—",
            "decision": f"Coluna {target_col} com {n_mis} ausências — modelo logístico não exigido",
        }

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    valid_preds = [c for c in predictor_cols if c != target_col and c in df.columns]
    X_raw = df[valid_preds].apply(pd.to_numeric, errors="coerce")
    X_imputed = X_raw.fillna(X_raw.median())
    X_std = StandardScaler().fit_transform(X_imputed.to_numpy(dtype=float))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(penalty=None, max_iter=500)
        clf.fit(X_std, y)

    p_hat = np.clip(clf.predict_proba(X_std)[:, 1], 1e-9, 1 - 1e-9)
    p_null = np.clip(y.mean(), 1e-9, 1 - 1e-9)

    ll_model = float(np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)))
    ll_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
    lr_stat = max(0.0, 2.0 * (ll_model - ll_null))
    dof = len(valid_preds)
    p_val = float(1.0 - stats.chi2.cdf(lr_stat, dof))

    coefs = np.abs(clf.coef_[0])
    top_pred = valid_preds[int(np.argmax(coefs))] if valid_preds else "—"
    decision = (
        f"Rejeita MCAR (χ² = {lr_stat:.2f}, p = {p_val:.4f} < 0,05; preditor dominante: {top_pred} → MAR)"
        if p_val < 0.05
        else f"Não rejeita MCAR (χ² = {lr_stat:.2f}, p = {p_val:.4f} ≥ 0,05)"
    )

    return {
        "applicable": True,
        "target_col": target_col,
        "statistic": round(lr_stat, 3),
        "df": dof,
        "p_value": round(p_val, 4),
        "top_predictor": top_pred,
        "decision": decision,
    }

