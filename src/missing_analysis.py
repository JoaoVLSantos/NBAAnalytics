"""
missing_analysis.py
===================
Orquestrador de diagnóstico de valores em falta (missing values), explicação dos
mecanismos clássicos (MCAR, MAR, MNAR) e imputação.

Cada teste estatístico individual reside no seu próprio ficheiro em `src/methods/`:
    * src/methods/little_test.py          -> Teste de Little (1988)
    * src/methods/jamshidian_jalal.py     -> Teste de Jamshidian-Jalal (2010)
    * src/methods/listing_schlittgen.py   -> Teste de Listing e Schlittgen (LS, 1998)
    * src/methods/fairclough_logistic.py  -> Regressão Logística de Fairclough (1998/2002)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .methods.fairclough_logistic import FAIRCLOUGH_LOGISTIC_GUIDE, fairclough_logistic_test
from .methods.jamshidian_jalal import JAMSHIDIAN_TEST_GUIDE, jamshidian_jalal_test
from .methods.listing_schlittgen import LISTING_SCHLITTGEN_GUIDE, listing_schlittgen_test
from .methods.little_test import LITTLE_TEST_GUIDE, littles_mcar_test

# ------------------------------------------------------------------
# Referencial teórico dos mecanismos MCAR, MAR e MNAR
# ------------------------------------------------------------------
MISSING_MECHANISMS: list[dict[str, str]] = [
    {
        "code": "MCAR",
        "title": "Missing Completely at Random (Ausente Completamente ao Acaso)",
        "formula": "P(R = 0 | Y_obs, Y_mis) = P(R = 0)",
        "description": (
            "A probabilidade de um registo estar ausente é igual para todas as observações e "
            "independe tanto dos valores observados (Y_obs) quanto dos valores não observados (Y_mis). "
            "O processo gerador da perda de dados é puramente aleatório."
        ),
        "nba_example": (
            "Uma falha momentânea de transmissão apaga aleatoriamente o registo de ressaltos (REB) "
            "de algumas partidas, sem qualquer relação com o time, pontuação ou tempo de jogo."
        ),
        "consequence": (
            "A remoção de linhas incompletas (listwise deletion) não introduz viés nas estimativas, "
            "reduzindo apenas o tamanho amostral e a potência estatística."
        ),
    },
    {
        "code": "MAR",
        "title": "Missing at Random (Ausente ao Acaso Condicional)",
        "formula": "P(R = 0 | Y_obs, Y_mis) = P(R = 0 | Y_obs)",
        "description": (
            "A probabilidade de ausência depende de outras variáveis observadas no conjunto de dados "
            "(Y_obs), mas NÃO depende do próprio valor ausente (Y_mis) após controlar por Y_obs."
        ),
        "nba_example": (
            "O registo de desarmes (BLK) ou roubos de bola (STL) faltar com maior frequência em jogos "
            "com menos minutos disputados (MIN) ou de determinadas franquias, mas não devido à quantidade "
            "de desarmes em si."
        ),
        "consequence": (
            "A simples exclusão de linhas gera viés. Deve-se recorrer a imputação condicional baseada "
            "nas variáveis observadas (MICE, regressão, KNN) ou máxima verossimilhança."
        ),
    },
    {
        "code": "MNAR",
        "title": "Missing Not at Random (Ausente Não ao Acaso / Não Ignorável)",
        "formula": "P(R = 0 | Y_obs, Y_mis) depende de Y_mis",
        "description": (
            "A probabilidade de um valor estar em falta depende diretamente do próprio valor não observado "
            "(Y_mis), mesmo após condicionar em todas as variáveis observadas (Y_obs)."
        ),
        "nba_example": (
            "A eficiência de triplos (FG3_PCT) ser omitida exatamente nas partidas em que o aproveitamento "
            "foi extremamente baixo, ou estatísticas omitidas por lesão diretamente ligada ao desempenho."
        ),
        "consequence": (
            "Mecanismo não ignorável: imputações padrão produzem viés severo; requer modelação conjunta "
            "do mecanismo de ausência (modelos de seleção de Heckman ou pattern-mixture models)."
        ),
    },
]

MISSING_TESTS_GUIDE: list[dict[str, str]] = [
    LITTLE_TEST_GUIDE,
    JAMSHIDIAN_TEST_GUIDE,
    LISTING_SCHLITTGEN_GUIDE,
    FAIRCLOUGH_LOGISTIC_GUIDE,
]


# ------------------------------------------------------------------
# Funções de resumo e imputação
# ------------------------------------------------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem e percentagem de valores em falta por coluna (ordem decrescente)."""
    n = len(df)
    summary = pd.DataFrame({
        "missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2) if n else 0.0,
        "dtype": df.dtypes.astype(str),
    })
    return summary.sort_values("missing", ascending=False)


def missing_by_group(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """Percentagem de missing por grupo (ex.: por Time) — útil para avaliar MCAR vs MAR."""
    return (df.groupby(group_col)[cols]
              .apply(lambda g: g.isna().mean() * 100)
              .round(2))


def impute(df: pd.DataFrame, cols: list[str], strategy: str = "median") -> pd.DataFrame:
    """Imputação simples (média/mediana/zero). Devolve uma cópia."""
    out = df.copy()
    for col in cols:
        if strategy == "median":
            fill = out[col].median()
        elif strategy == "mean":
            fill = out[col].mean()
        elif strategy == "zero":
            fill = 0
        else:
            raise ValueError(f"Estratégia desconhecida: {strategy}")
        out[col] = out[col].fillna(fill)
    return out


def evaluate_missing_tests(df: pd.DataFrame, cols: list[str], target_col: str, sim_mode: str = "real") -> dict:
    """Executa os 4 testes de `src/methods/` na base real ou num cenário simulado (MCAR vs MAR)."""
    work_df = df.copy()
    rng = np.random.default_rng(42)

    if sim_mode == "mcar":
        for c in cols:
            mask = rng.random(len(work_df)) < 0.06
            work_df.loc[mask, c] = np.nan
    elif sim_mode == "mar":
        aux = (
            pd.to_numeric(work_df["REB"], errors="coerce").fillna(0)
            + pd.to_numeric(work_df["AST"], errors="coerce").fillna(0)
        )
        prob_missing = np.where(aux > aux.quantile(0.70), 0.25, 0.02)
        mask_target = rng.random(len(work_df)) < prob_missing
        work_df.loc[mask_target, target_col] = np.nan

    aux_cols = [c for c in cols if c != target_col]
    return {
        "sim_mode": sim_mode,
        "little": littles_mcar_test(work_df, cols),
        "jamshidian": jamshidian_jalal_test(work_df, cols),
        "listing_schlittgen": listing_schlittgen_test(work_df, target_col, aux_cols),
        "fairclough": fairclough_logistic_test(work_df, target_col, aux_cols),
        "work_df": work_df,
    }
