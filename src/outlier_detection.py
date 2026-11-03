"""
outlier_detection.py
====================
Orquestrador de deteção de valores inconsistentes e outliers.

Cada método individual reside no seu próprio ficheiro dentro de `src/methods/`:
    * src/methods/iqr_boxplot.py  -> IQR / Boxplot e extremos
    * src/methods/zscore.py       -> Z-Score Clássico e Z-Score Modificado (MAD)
    * src/methods/grubbs.py       -> Teste de Grubbs (ESD, 1950)
    * src/methods/ecod.py         -> ECOD (Li et al., 2022)
    * src/methods/knn.py          -> K-NN para anomalias (Ramaswamy et al., 2000)
    * src/methods/dbscan.py       -> DBSCAN (Ester et al., 1996)
    * src/methods/lof.py          -> LOF (Breunig et al., 2000)
"""

from __future__ import annotations

import pandas as pd

from .methods.dbscan import DBSCAN_THEORY, detect_outliers_dbscan
from .methods.ecod import ECOD_THEORY, detect_outliers_ecod
from .methods.grubbs import GRUBBS_THEORY, detect_outliers_grubbs, grubbs_test_summary
from .methods.iqr_boxplot import IQR_THEORY, detect_outliers_iqr, iqr_bounds
from .methods.knn import KNN_THEORY, detect_outliers_knn
from .methods.lof import LOF_THEORY, detect_outliers_lof
from .methods.zscore import ZSCORE_THEORY, detect_outliers_modified_zscore, detect_outliers_zscore

OUTLIER_THEORY_GUIDE: list[dict[str, str]] = [
    IQR_THEORY,
    ZSCORE_THEORY,
    GRUBBS_THEORY,
    ECOD_THEORY,
    KNN_THEORY,
    DBSCAN_THEORY,
    LOF_THEORY,
]


def detect_outliers_pyod(
    df: pd.DataFrame, cols: list[str], contamination: float = 0.05, method: str = "ecod"
) -> pd.Series:
    """Interface unificada delegando para os módulos em `src/methods/`."""
    if method == "ecod":
        return detect_outliers_ecod(df, cols, contamination=contamination)
    if method == "knn":
        return detect_outliers_knn(df, cols, contamination=contamination)
    if method == "lof":
        return detect_outliers_lof(df, cols, contamination=contamination)
    if method == "dbscan":
        return detect_outliers_dbscan(df, cols)

    from pyod.models.iforest import IForest
    from sklearn.preprocessing import StandardScaler

    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 2:
        return pd.Series(False, index=df.index)
    X = StandardScaler().fit_transform(data.to_numpy(dtype=float))
    model = IForest(contamination=contamination, random_state=42)
    model.fit(X)
    labels = pd.Series(False, index=df.index)
    labels.loc[data.index] = model.labels_.astype(bool)
    return labels


def summarize_columns_inconsistencies_and_outliers(
    raw_df: pd.DataFrame, clean_df: pd.DataFrame, dictionary: dict[str, dict[str, str]]
) -> list[dict]:
    """Calcula, para CADA coluna da base, a quantidade de campos inconsistentes e de outliers."""
    raw_norm = raw_df.copy()
    raw_norm.columns = [str(c).strip().upper().replace(" ", "_") for c in raw_norm.columns]

    pct_cols = {"W_PCT", "FG_PCT", "FG3_PCT", "FT_PCT"}
    id_or_cat_cols = {"TEAM_ID", "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "TEAM_NAME", "MATCHUP", "WL"}

    summary_rows: list[dict] = []

    for col in clean_df.columns:
        meta = dictionary.get(col, {})
        label = meta.get("label", col)
        col_type = meta.get("type", str(clean_df[col].dtype))

        # 1. Contagem de valores inconsistentes no arquivo bruto
        inconsistent = 0
        inconsistency_note = "Nenhuma inconsistência"
        if col in raw_norm.columns:
            raw_series = raw_norm[col]
            if col in pct_cols:
                raw_num = pd.to_numeric(raw_series, errors="coerce")
                n_scaled = int(((raw_num > 1.0) | (raw_num < 0.0)).sum())
                n_non_num = int(raw_num.isna().sum() - raw_series.isna().sum())
                inconsistent = n_scaled + n_non_num
                if n_scaled > 0:
                    inconsistency_note = f"{n_scaled} valores > 1,0 (escala ×1000)"
            elif col == "WL":
                raw_num = pd.to_numeric(raw_series, errors="coerce")
                inconsistent = int((~raw_num.isin([0, 1])).sum())
            elif col not in id_or_cat_cols:
                raw_num = pd.to_numeric(raw_series, errors="coerce")
                n_neg = int((raw_num < 0).sum())
                n_non_num = int(raw_num.isna().sum() - raw_series.isna().sum())
                inconsistent = n_neg + n_non_num
                if inconsistent > 0:
                    inconsistency_note = f"{inconsistent} negativos/inválidos"

        # 2. Para colunas quantitativas (exceto IDs/binárias), calcula outliers nos métodos de `src/methods/`
        is_quantitative = (col not in id_or_cat_cols) and pd.api.types.is_numeric_dtype(clean_df[col])
        if is_quantitative:
            s = clean_df[col]
            col_df = clean_df[[col]]
            n_iqr_mod = int(detect_outliers_iqr(s, k=1.5).sum())
            n_iqr_ext = int(detect_outliers_iqr(s, k=3.0).sum())
            n_zscore = int(detect_outliers_zscore(s, threshold=3.0).sum())
            n_mod_z = int(detect_outliers_modified_zscore(s, threshold=3.5).sum())
            n_grubbs = int(detect_outliers_grubbs(s, alpha=0.05).sum())
            n_ecod = int(detect_outliers_ecod(col_df, [col], contamination=0.05).sum())
            n_knn = int(detect_outliers_knn(col_df, [col], contamination=0.05).sum())
            n_dbscan = int(detect_outliers_dbscan(col_df, [col]).sum())
            n_lof = int(detect_outliers_lof(col_df, [col], contamination=0.05).sum())
        else:
            n_iqr_mod = n_iqr_ext = n_zscore = n_mod_z = n_grubbs = n_ecod = n_knn = n_dbscan = n_lof = None

        summary_rows.append({
            "column": col,
            "label": label,
            "type": col_type,
            "is_quantitative": is_quantitative,
            "inconsistent": inconsistent,
            "inconsistency_note": inconsistency_note,
            "iqr_mod": n_iqr_mod,
            "iqr_ext": n_iqr_ext,
            "zscore": n_zscore,
            "mod_zscore": n_mod_z,
            "grubbs": n_grubbs,
            "ecod": n_ecod,
            "knn": n_knn,
            "dbscan": n_dbscan,
            "lof": n_lof,
        })

    return summary_rows


__all__ = [
    "OUTLIER_THEORY_GUIDE",
    "iqr_bounds",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "detect_outliers_modified_zscore",
    "grubbs_test_summary",
    "detect_outliers_grubbs",
    "detect_outliers_ecod",
    "detect_outliers_knn",
    "detect_outliers_dbscan",
    "detect_outliers_lof",
    "detect_outliers_pyod",
    "summarize_columns_inconsistencies_and_outliers",
]
