"""
app.py — Camada de apresentação (Flask) do Projeto 1 LCD: NBA Analytics.

Responsabilidades deste ficheiro:
    * Inicializar a aplicação e carregar os dados (uma vez, no arranque).
    * Receber pedidos HTTP, validar filtros e delegar os cálculos à camada
      analítica (`src/`).
    * Construir as figuras Plotly e enviá-las em JSON para os templates.

Nenhum cálculo estatístico é feito aqui — tudo vive em `src/`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template, request

from src.data_prep import (
    DATA_DICTIONARY,
    PROCESSED_PATH,
    TARGET_VARS,
    TEAM_COL,
    load_dataset,
    load_raw_dataset,
)
from src.missing_analysis import (
    MISSING_MECHANISMS,
    MISSING_TESTS_GUIDE,
    evaluate_missing_tests,
    impute,
    missing_by_group,
    missing_summary,
)
from src.outlier_detection import (
    OUTLIER_THEORY_GUIDE,
    detect_outliers_dbscan,
    detect_outliers_ecod,
    detect_outliers_grubbs,
    detect_outliers_iqr,
    detect_outliers_knn,
    detect_outliers_lof,
    detect_outliers_modified_zscore,
    detect_outliers_zscore,
    grubbs_test_summary,
    iqr_bounds,
    summarize_columns_inconsistencies_and_outliers,
)
from src.stats_metrics import calculate_univariate_stats, interpret_cv, interpret_skewness

# ------------------------------------------------------------------
# Configuração
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-change-me")

# Carregamento único dos dados (bruto e processado)
RAW_DF = load_raw_dataset()
DF = load_dataset(PROCESSED_PATH)

# Variáveis-alvo (apenas as que de facto existem no dataset carregado)
TARGET_VARIABLES: list[str] = [v for v in TARGET_VARS if v in DF.columns]
TEAMS: list[str] = sorted(DF[TEAM_COL].dropna().unique().tolist())
ALL_TEAMS = "ALL"

# Resumo por coluna de valores inconsistentes e outliers (calculado uma vez no arranque)
COLUMN_OUTLIER_SUMMARY = summarize_columns_inconsistencies_and_outliers(RAW_DF, DF, DATA_DICTIONARY)
QUANTITATIVE_VARIABLES: list[str] = TARGET_VARIABLES + [
    r["column"] for r in COLUMN_OUTLIER_SUMMARY
    if r["is_quantitative"] and r["column"] not in TARGET_VARIABLES
]

ACCENT = "#f97316"  # cor de destaque (laranja "bola de basquetebol")

# Estrutura da tabela descritiva: (grupo, [(chave, rótulo, notação)])
STAT_GROUPS = [
    ("Tendência Central", [
        ("mean", "Média", "x̄"),
        ("median", "Mediana", "Md"),
        ("mode", "Moda", "Mo"),
    ]),
    ("Posição", [
        ("min", "Mínimo", "min"),
        ("q1", "1.º Quartil", "Q1"),
        ("q3", "3.º Quartil", "Q3"),
        ("max", "Máximo", "max"),
    ]),
    ("Dispersão", [
        ("range", "Amplitude", "R"),
        ("iqr", "Amplitude Interquartil", "IQR"),
        ("variance", "Variância Amostral", "s²"),
        ("std", "Desvio Padrão Amostral", "s"),
        ("cv_pct", "Coeficiente de Variação", "CV%"),
    ]),
    ("Forma", [
        ("skewness", "Assimetria (Fisher-Pearson ajust.)", "G₁"),
    ]),
]


# ------------------------------------------------------------------
# Filtros e contexto global Jinja2
# ------------------------------------------------------------------
@app.template_filter("num")
def format_number(value, decimals: int = 2) -> str:
    """Formata números no padrão português (espaço nos milhares, vírgula decimal)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}".replace(",", " ")
    try:
        text = f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)
    return text.replace(",", " ").replace(".", ",")


# ------------------------------------------------------------------
# Helpers de visualização (apresentação)
# ------------------------------------------------------------------
def build_distribution_figure(data: pd.Series, variable: str, stats: dict) -> str:
    """Histograma + Boxplot marginal interativos, devolvidos em JSON."""
    label = DATA_DICTIONARY.get(variable, {}).get("label", variable)
    frame = data.dropna().to_frame(name=variable)

    fig = px.histogram(
        frame,
        x=variable,
        marginal="box",                 # boxplot sincronizado sobre o histograma
        nbins=30,
        color_discrete_sequence=[ACCENT],
        labels={variable: f"{label} ({variable})"},
        template="plotly_dark",
        opacity=0.85,
    )

    # Linhas de referência: média (tracejada) e mediana (pontilhada)
    if stats.get("mean") is not None:
        fig.add_vline(x=stats["mean"], line_dash="dash", line_color="#38bdf8", line_width=2)
    if stats.get("median") is not None:
        fig.add_vline(x=stats["median"], line_dash="dot", line_color="#a3e635", line_width=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#e2e8f0"},
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        bargap=0.06,
        yaxis_title="Frequência absoluta",
        hoverlabel={"font_family": "Inter, sans-serif"},
    )
    return pio.to_json(fig)


def build_quality_figure(scale_fixes: list[dict]) -> str:
    """Gráfico de barras comparando registos intactos vs. corrigidos nas colunas de proporção."""
    frame = pd.DataFrame(scale_fixes)
    fig = px.bar(
        frame,
        x="column",
        y=["valid_original", "fixed_count"],
        barmode="stack",
        labels={
            "column": "Coluna de Proporção",
            "value": "Número de Registos",
            "variable": "Estado no CSV Bruto",
        },
        color_discrete_map={"valid_original": "#38bdf8", "fixed_count": ACCENT},
        template="plotly_dark",
    )
    newnames = {"valid_original": "Escala original correta (0–1)", "fixed_count": "Corrigidos na limpeza (÷ 1000)"}
    fig.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#e2e8f0"},
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hoverlabel={"font_family": "Inter, sans-serif"},
    )
    return pio.to_json(fig)


def build_outliers_figure(subset: pd.DataFrame, variable: str, mask: pd.Series, method: str) -> str:
    """Gráfico de dispersão temporal + boxplot lateral destacando os outliers detetados."""
    label = DATA_DICTIONARY.get(variable, {}).get("label", variable)
    plot_df = subset.copy()
    plot_df["Status"] = np.where(mask, "Outlier", "Normal")
    x_col = "GAME_DATE" if "GAME_DATE" in plot_df.columns else plot_df.index

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=variable,
        color="Status",
        marginal_y="box",
        color_discrete_map={"Normal": "#38bdf8", "Outlier": ACCENT},
        hover_data=[c for c in ["TEAM_ABBREVIATION", "MATCHUP", "PTS", "REB", "AST"] if c in plot_df.columns],
        labels={variable: f"{label} ({variable})", "GAME_DATE": "Data do Jogo", "Status": "Classificação"},
        template="plotly_dark",
        opacity=0.85,
    )

    if method in ("iqr", "iqr_ext"):
        lower_mod, upper_mod = iqr_bounds(subset[variable].dropna(), k=1.5)
        lower_ext, upper_ext = iqr_bounds(subset[variable].dropna(), k=3.0)
        fig.add_hline(y=lower_mod, line_dash="dash", line_color="#a3e635", line_width=1.5, row=1, col=1)
        fig.add_hline(y=upper_mod, line_dash="dash", line_color="#f87171", line_width=1.5, row=1, col=1)
        fig.add_hline(y=lower_ext, line_dash="dot", line_color="#facc15", line_width=1.2, row=1, col=1)
        fig.add_hline(y=upper_ext, line_dash="dot", line_color="#facc15", line_width=1.2, row=1, col=1)
    elif method == "zscore":
        s_clean = subset[variable].dropna()
        if len(s_clean) > 1:
            m_val, std_val = float(s_clean.mean()), float(s_clean.std(ddof=1))
            fig.add_hline(y=m_val - 3 * std_val, line_dash="dash", line_color="#a3e635", line_width=1.5, row=1, col=1)
            fig.add_hline(y=m_val + 3 * std_val, line_dash="dash", line_color="#f87171", line_width=1.5, row=1, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#e2e8f0"},
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hoverlabel={"font_family": "Inter, sans-serif"},
    )
    return pio.to_json(fig)


# ------------------------------------------------------------------
# Rotas
# ------------------------------------------------------------------
@app.route("/")
def index():
    """Resumo da base de dados e dicionário de metadados."""
    total_cells = DF.size or 1
    summary = {
        "n_raw_rows": len(RAW_DF),
        "n_raw_cols": RAW_DF.shape[1],
        "n_rows": len(DF),
        "n_cols": DF.shape[1],
        "n_teams": len(TEAMS),
        "n_games": int(DF["GAME_ID"].nunique()) if "GAME_ID" in DF else None,
        "n_missing": int(DF.isna().sum().sum()),
        "pct_missing": DF.isna().sum().sum() / total_cells * 100,
    }

    # Dicionário apenas com as colunas presentes no dataset
    dictionary = [
        {"column": col, **meta, "is_target": col in TARGET_VARIABLES}
        for col, meta in DATA_DICTIONARY.items() if col in DF.columns
    ]

    # Pré-visualização do ficheiro bruto e do ficheiro tratado (NaN -> None para o Jinja2)
    raw_preview = RAW_DF.head(8).astype(object).where(RAW_DF.head(8).notna(), None)
    preview = DF.head(8).astype(object).where(DF.head(8).notna(), None)

    return render_template(
        "index.html",
        summary=summary,
        dictionary=dictionary,
        raw_preview_cols=raw_preview.columns.tolist(),
        raw_preview_rows=raw_preview.values.tolist(),
        preview_cols=preview.columns.tolist(),
        preview_rows=preview.values.tolist(),
        target_vars=TARGET_VARIABLES,
    )


@app.route("/descritiva", methods=["GET", "POST"])
def descritiva():
    """Estatística descritiva univariada filtrada por Time e Variável."""
    params = request.form if request.method == "POST" else request.args

    # Validação defensiva: só aceitamos valores conhecidos
    team = params.get("team", ALL_TEAMS)
    if team != ALL_TEAMS and team not in TEAMS:
        team = ALL_TEAMS
    variable = params.get("variable", TARGET_VARIABLES[0])
    if variable not in TARGET_VARIABLES:
        variable = TARGET_VARIABLES[0]

    subset = DF if team == ALL_TEAMS else DF[DF[TEAM_COL] == team]
    series = subset[variable]

    # Camada analítica
    stats = calculate_univariate_stats(series)
    graph_json = build_distribution_figure(series, variable, stats) if stats["n"] else None

    return render_template(
        "descriptive.html",
        teams=TEAMS,
        variables=TARGET_VARIABLES,
        dictionary=DATA_DICTIONARY,
        selected_team=team,
        selected_variable=variable,
        stats=stats,
        stat_groups=STAT_GROUPS,
        skew_text=interpret_skewness(stats["skewness"]),
        cv_text=interpret_cv(stats["cv_pct"]),
        graph_json=graph_json,
    )


@app.route("/qualidade", methods=["GET", "POST"])
def qualidade():
    """Diagnóstico de qualidade dos dados, valores em falta, mecanismos e testes estatísticos."""
    params = request.form if request.method == "POST" else request.args
    strategy = params.get("strategy", "median")
    if strategy not in ("median", "mean", "zero"):
        strategy = "median"
    variable = params.get("variable", TARGET_VARIABLES[0])
    if variable not in TARGET_VARIABLES:
        variable = TARGET_VARIABLES[0]
    sim_mode = params.get("sim_mode", "real")
    if sim_mode not in ("real", "mcar", "mar"):
        sim_mode = "real"

    # 1. Resumo de missing por coluna na base oficial (src/missing_analysis.py)
    ms_df = missing_summary(DF)
    missing_rows = [
        {
            "column": col,
            "label": DATA_DICTIONARY.get(col, {}).get("label", col),
            "dtype": row["dtype"],
            "valid": int(len(DF) - row["missing"]),
            "missing": int(row["missing"]),
            "pct_missing": float(row["pct_missing"]),
        }
        for col, row in ms_df.iterrows()
    ]

    # 2. Execução dos 4 testes estatísticos (Little, Jamshidian-Jalal, Listing-Schlittgen, Fairclough)
    test_results = evaluate_missing_tests(DF, TARGET_VARIABLES, variable, sim_mode=sim_mode)
    active_df = test_results["work_df"]

    # 3. Missing por Time nas variáveis-alvo
    group_ms = missing_by_group(active_df, TEAM_COL, TARGET_VARIABLES)
    group_rows = [
        {"team": team, "values": [float(group_ms.loc[team, c]) for c in TARGET_VARIABLES]}
        for team in group_ms.index
    ]

    # 4. Demonstração de imputação (antes vs. depois sobre o dataset selecionado)
    imputed_df = impute(active_df, TARGET_VARIABLES, strategy=strategy)
    stats_before = calculate_univariate_stats(active_df[variable])
    stats_after = calculate_univariate_stats(imputed_df[variable])

    # 5. Diagnóstico de limpeza entre o CSV Bruto e o CSV Tratado
    pct_cols = [c for c in ["W_PCT", "FG_PCT", "FG3_PCT", "FT_PCT"] if c in RAW_DF.columns]
    scale_fixes = []
    total_scale_fixes = 0
    for c in pct_cols:
        raw_num = pd.to_numeric(RAW_DF[c], errors="coerce")
        fixed = int((raw_num > 1.0).sum())
        total_scale_fixes += fixed
        scale_fixes.append({
            "column": c,
            "label": DATA_DICTIONARY.get(c, {}).get("label", c),
            "fixed_count": fixed,
            "valid_original": int(len(RAW_DF) - fixed),
            "pct_fixed": round(fixed / len(RAW_DF) * 100, 1) if len(RAW_DF) else 0.0,
        })

    quality_kpis = {
        "total_cells": int(DF.size),
        "total_missing": int(DF.isna().sum().sum()),
        "pct_completeness": round((1 - DF.isna().sum().sum() / (DF.size or 1)) * 100, 2),
        "duplicates_removed": int(len(RAW_DF) - len(RAW_DF.drop_duplicates())),
        "total_scale_fixes": total_scale_fixes,
        "new_cols_created": int(DF.shape[1] - RAW_DF.shape[1]),
    }

    graph_json = build_quality_figure(scale_fixes) if scale_fixes else None

    return render_template(
        "data_cleaning.html",
        quality_kpis=quality_kpis,
        missing_rows=missing_rows,
        scale_fixes=scale_fixes,
        group_rows=group_rows,
        mechanisms=MISSING_MECHANISMS,
        tests_guide=MISSING_TESTS_GUIDE,
        test_results=test_results,
        variables=TARGET_VARIABLES,
        dictionary=DATA_DICTIONARY,
        selected_strategy=strategy,
        selected_variable=variable,
        selected_sim_mode=sim_mode,
        stats_before=stats_before,
        stats_after=stats_after,
        graph_json=graph_json,
    )


OUTLIER_METHODS = {
    "iqr": "IQR / Boxplot Moderado (k = 1,5)",
    "iqr_ext": "IQR / Boxplot Extremo (k = 3,0)",
    "zscore": "Z-Score Clássico (|z| > 3,0)",
    "mod_zscore": "Z-Score Modificado / MAD (|Mz| > 3,5)",
    "grubbs": "Teste de Grubbs (ESD Sequencial, α = 5%)",
    "ecod": "ECOD · Distribuição Acumulada Empírica (5%)",
    "knn": "K-NN · K-Nearest Neighbors (k = 5, 5%)",
    "dbscan": "DBSCAN · Agrupamento por Densidade (Ruído)",
    "lof": "LOF · Local Outlier Factor (k = 20, 5%)",
}


@app.route("/outliers", methods=["GET", "POST"])
def outliers():
    """Deteção de valores inconsistentes e outliers univariados e multivariados."""
    params = request.form if request.method == "POST" else request.args

    team = params.get("team", ALL_TEAMS)
    if team != ALL_TEAMS and team not in TEAMS:
        team = ALL_TEAMS
    variable = params.get("variable", QUANTITATIVE_VARIABLES[0])
    if variable not in QUANTITATIVE_VARIABLES:
        variable = QUANTITATIVE_VARIABLES[0]
    method = params.get("method", "iqr")
    if method not in OUTLIER_METHODS:
        method = "iqr"

    subset = DF if team == ALL_TEAMS else DF[DF[TEAM_COL] == team]
    series = subset[variable]

    # Calcula as máscaras para todos os métodos exigidos para comparação direta
    masks = {
        "iqr": detect_outliers_iqr(series, k=1.5),
        "iqr_ext": detect_outliers_iqr(series, k=3.0),
        "zscore": detect_outliers_zscore(series, threshold=3.0),
        "mod_zscore": detect_outliers_modified_zscore(series, threshold=3.5),
        "grubbs": detect_outliers_grubbs(series, alpha=0.05),
        "ecod": detect_outliers_ecod(subset, TARGET_VARIABLES, contamination=0.05),
        "knn": detect_outliers_knn(subset, TARGET_VARIABLES, contamination=0.05),
        "dbscan": detect_outliers_dbscan(subset, TARGET_VARIABLES),
        "lof": detect_outliers_lof(subset, TARGET_VARIABLES, contamination=0.05),
    }

    active_mask = masks[method]
    lower_iqr, upper_iqr = iqr_bounds(series.dropna(), k=1.5) if not series.dropna().empty else (None, None)
    lower_ext, upper_ext = iqr_bounds(series.dropna(), k=3.0) if not series.dropna().empty else (None, None)
    grubbs_info = grubbs_test_summary(series, alpha=0.05)

    method_counts = [
        {
            "key": k,
            "label": label,
            "count": int(m.sum()),
            "pct": round(float(m.mean() * 100), 2) if len(m) else 0.0,
        }
        for k, (label, m) in zip(OUTLIER_METHODS.keys(), zip(OUTLIER_METHODS.values(), masks.values()))
    ]

    # Registos identificados como outliers pelo método ativo
    outlier_df = subset[active_mask].sort_values(by=variable, ascending=False)
    extra_var = [variable] if variable not in TARGET_VARIABLES else []
    display_cols = [c for c in ["GAME_DATE", TEAM_COL, "MATCHUP", "WL"] + extra_var + TARGET_VARIABLES if c in outlier_df.columns]
    outlier_rows = (
        outlier_df[display_cols]
        .head(50)
        .astype(object)
        .where(outlier_df[display_cols].head(50).notna(), None)
        .values.tolist()
    )

    graph_json = build_outliers_figure(subset, variable, active_mask, method) if not subset.empty else None

    return render_template(
        "outliers.html",
        teams=TEAMS,
        variables=QUANTITATIVE_VARIABLES,
        methods=OUTLIER_METHODS,
        dictionary=DATA_DICTIONARY,
        selected_team=team,
        selected_variable=variable,
        selected_method=method,
        method_counts=method_counts,
        column_summary=COLUMN_OUTLIER_SUMMARY,
        theory_guide=OUTLIER_THEORY_GUIDE,
        grubbs_info=grubbs_info,
        n_total=len(subset),
        n_outliers=int(active_mask.sum()),
        pct_outliers=round(float(active_mask.mean() * 100), 2) if len(subset) else 0.0,
        lower_iqr=lower_iqr,
        upper_iqr=upper_iqr,
        lower_ext=lower_ext,
        upper_ext=upper_ext,
        outlier_cols=display_cols,
        outlier_rows=outlier_rows,
        graph_json=graph_json,
    )


@app.route("/teoria")
def teoria():
    """Explicações e base teórica dos mecanismos de missing, testes e métodos de outliers."""
    return render_template(
        "theory.html",
        mechanisms=MISSING_MECHANISMS,
        tests_guide=MISSING_TESTS_GUIDE,
        theory_guide=OUTLIER_THEORY_GUIDE,
    )


@app.errorhandler(404)
def not_found(_error):
    return render_template("base.html", not_found=True), 404


# ------------------------------------------------------------------
# Execução local (em produção: gunicorn app:app)
# ------------------------------------------------------------------
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=debug)
