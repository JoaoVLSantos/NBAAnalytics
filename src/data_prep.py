from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Caminhos e constantes do projeto
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_PATH = RAW_DIR / "stats_times_regular_final - stats_times_regular_final.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "nba_cleaned.csv"

# Variáveis-alvo da análise (box score por jogo)
TARGET_VARS: list[str] = ["PTS", "REB", "AST", "STL", "BLK", "PF", "FG3M"]

TEAM_COL = "TEAM_ABBREVIATION"

# Dicionário de metadados: coluna -> descrição, tipo e unidade
DATA_DICTIONARY: dict[str, dict[str, str]] = {
    "TEAM_ID":           {"label": "ID do Time",        "desc": "Identificador único da franquia NBA",            "type": "Categórica nominal", "unit": "—"},
    "GAME_ID":           {"label": "ID do Jogo",        "desc": "Identificador único da partida",                 "type": "Categórica nominal", "unit": "—"},
    "GAME_DATE":         {"label": "Data do Jogo",      "desc": "Data em que a partida foi disputada",            "type": "Temporal",           "unit": "AAAA/MM/DD"},
    "TEAM_ABBREVIATION": {"label": "Sigla do Time",     "desc": "Sigla oficial de 3 letras da franquia NBA",      "type": "Categórica nominal", "unit": "—"},
    "TEAM_NAME":         {"label": "Nome do Time",      "desc": "Nome completo da franquia NBA",                  "type": "Categórica nominal", "unit": "—"},
    "MATCHUP":           {"label": "Confronto",         "desc": "Equipas e mando de campo (vs. casa / @ fora)",   "type": "Categórica nominal", "unit": "—"},
    "WL":                {"label": "Resultado (V/D)",   "desc": "Indicador de vitória (1) ou derrota (0)",        "type": "Categórica binária", "unit": "0 / 1"},
    "W":                 {"label": "Vitórias Acum.",    "desc": "Vitórias acumuladas na temporada até ao jogo",   "type": "Quantitativa discreta", "unit": "vitórias"},
    "L":                 {"label": "Derrotas Acum.",    "desc": "Derrotas acumuladas na temporada até ao jogo",   "type": "Quantitativa discreta", "unit": "derrotas"},
    "W_PCT":             {"label": "Aproveitamento",    "desc": "Proporção de vitórias na temporada",             "type": "Quantitativa contínua", "unit": "proporção (0–1)"},
    "MIN":               {"label": "Minutos",           "desc": "Minutos totais disputados na partida",           "type": "Quantitativa contínua", "unit": "min"},
    "FGM":               {"label": "Arremessos Conv.",  "desc": "Arremessos de quadra convertidos",               "type": "Quantitativa discreta", "unit": "contagem"},
    "FGA":               {"label": "Arremessos Tent.",  "desc": "Arremessos de quadra tentados",                  "type": "Quantitativa discreta", "unit": "contagem"},
    "FG_PCT":            {"label": "% Arremessos Quadra", "desc": "Eficiência nos arremessos de quadra (FGM/FGA)", "type": "Quantitativa contínua", "unit": "proporção (0–1)"},
    "FG3M":              {"label": "Triplos Convertidos", "desc": "Lançamentos de 3 pontos convertidos",          "type": "Quantitativa discreta", "unit": "contagem"},
    "FG3A":              {"label": "Triplos Tentados",  "desc": "Lançamentos de 3 pontos tentados",               "type": "Quantitativa discreta", "unit": "contagem"},
    "FG3_PCT":           {"label": "% Triplos",         "desc": "Eficiência nos lançamentos de 3 pontos",         "type": "Quantitativa contínua", "unit": "proporção (0–1)"},
    "FTM":               {"label": "Lances Livres Conv.", "desc": "Lances livres convertidos",                    "type": "Quantitativa discreta", "unit": "contagem"},
    "FTA":               {"label": "Lances Livres Tent.", "desc": "Lances livres tentados",                       "type": "Quantitativa discreta", "unit": "contagem"},
    "FT_PCT":            {"label": "% Lances Livres",   "desc": "Eficiência nos lances livres (FTM/FTA)",         "type": "Quantitativa contínua", "unit": "proporção (0–1)"},
    "OREB":              {"label": "Ressaltos Ofensivos", "desc": "Ressaltos conquistados no ataque",             "type": "Quantitativa discreta", "unit": "contagem"},
    "DREB":              {"label": "Ressaltos Defensivos", "desc": "Ressaltos conquistados na defesa",            "type": "Quantitativa discreta", "unit": "contagem"},
    "REB":               {"label": "Ressaltos",         "desc": "Ressaltos totais (ofensivos + defensivos)",      "type": "Quantitativa discreta", "unit": "contagem"},
    "AST":               {"label": "Assistências",      "desc": "Passes que resultaram diretamente em cesto",     "type": "Quantitativa discreta", "unit": "contagem"},
    "STL":               {"label": "Roubos de Bola",    "desc": "Recuperações de posse ao adversário",            "type": "Quantitativa discreta", "unit": "contagem"},
    "BLK":               {"label": "Desarmes de Lançamento", "desc": "Lançamentos adversários bloqueados",        "type": "Quantitativa discreta", "unit": "contagem"},
    "TOV":               {"label": "Perdas de Bola",    "desc": "Turnovers(perda de posse de bola) cometidos na partida",                 "type": "Quantitativa discreta", "unit": "contagem"},
    "PF":                {"label": "Faltas Pessoais",   "desc": "Faltas pessoais cometidas",                      "type": "Quantitativa discreta", "unit": "contagem"},
    "PTS":               {"label": "Pontos",            "desc": "Total de pontos marcados",                       "type": "Quantitativa discreta", "unit": "pontos"},
}

# ------------------------------------------------------------------
# Limpeza
# ------------------------------------------------------------------
def _find_raw_csv() -> Path | None:
    if RAW_PATH.is_file():
        return RAW_PATH
    if RAW_DIR.is_dir():
        csv_files = sorted(RAW_DIR.glob("*.csv"))
        if csv_files:
            return csv_files[0]
    return None


def _abbreviate_team_name(name: str) -> str:
    """Gera uma sigla de 3 letras a partir do nome da equipa.

    - 1 palavra:  3 primeiras letras (ex.: 'Lakers' -> 'LAK')
    - 2 palavras: 1.ª letra do 1.º nome + 2 primeiras do 2.º nome (ex.: 'Atlanta Hawks' -> 'AHA')
    - 3+ palavras: 1.ª letra de cada um dos 3 primeiros nomes (ex.: 'Los Angeles Lakers' -> 'LAL')
    """
    words = str(name).strip().split()
    if not words:
        return ""
    if len(words) == 1:
        return words[0][:3].upper()
    if len(words) == 2:
        return (words[0][0] + words[1][:2]).upper()
    return (words[0][0] + words[1][0] + words[2][0]).upper()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpeza básica e idempotente.

    1. Normaliza nomes de colunas (MAIÚSCULAS, sem espaços).
    2. Remove linhas duplicadas.
    3. Remove espaços em colunas de texto.
    4. Garante a coluna `TEAM_ABBREVIATION` (extraída de `MATCHUP` ou abreviada de `TEAM_NAME`).
    5. Converte as variáveis numéricas e corrige percentagens desformatadas (> 1 -> /1000).
    6. Valores negativos (impossíveis em contagens) passam a NaN.

    NOTA: a imputação de missing NÃO é feita aqui — é uma decisão
    analítica a tomar no módulo `missing_analysis`.
    """
    out = df.copy()
    out.columns = [str(c).strip().upper().replace(" ", "_") for c in out.columns]
    out = out.drop_duplicates().reset_index(drop=True)

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].str.strip()

    # Se TEAM_ABBREVIATION não existir no CSV original, extrai-a de MATCHUP ou gera sigla a partir de TEAM_NAME
    if TEAM_COL not in out.columns:
        if "MATCHUP" in out.columns:
            out[TEAM_COL] = out["MATCHUP"].astype(str).str.split().str[0]
        elif "TEAM_NAME" in out.columns:
            out[TEAM_COL] = out["TEAM_NAME"].map(_abbreviate_team_name)

    for col in [c for c in TARGET_VARS + ["MIN"] if c in out.columns]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] < 0, col] = np.nan

    # Corrige colunas de proporção onde decimais com 3 casas foram exportados como inteiros (ex.: 506 -> 0.506)
    for pct_col in ["W_PCT", "FG_PCT", "FG3_PCT", "FT_PCT"]:
        if pct_col in out.columns:
            out[pct_col] = pd.to_numeric(out[pct_col], errors="coerce")
            mask_scaled = out[pct_col] > 1.0
            out.loc[mask_scaled, pct_col] = (out.loc[mask_scaled, pct_col] / 1000.0).round(3)

    return out


# ------------------------------------------------------------------
# Carregamento e geração do CSV processado
# ------------------------------------------------------------------
def load_raw_dataset() -> pd.DataFrame:
    """Carrega o ficheiro CSV bruto original de `data/raw/` sem aplicar limpeza."""
    raw_file = _find_raw_csv()
    if raw_file is None:
        raise FileNotFoundError(f"Nenhum ficheiro CSV encontrado em {RAW_DIR}")
    return pd.read_csv(raw_file)


def load_dataset(path: Path | str = PROCESSED_PATH) -> pd.DataFrame:
    """Carrega o CSV processado; se não existir, gera-o (ou limpa em memória em ambientes read-only como Vercel)."""
    path = Path(path)
    if not path.is_file():
        try:
            build_processed_dataset()
        except OSError:
            # Em ambientes serverless (Vercel), o sistema de ficheiros é read-only
            df = clean_data(load_raw_dataset())
            logger.info("Dados processados em memória (%d linhas)", len(df))
            return df

    df = clean_data(pd.read_csv(path)) if path.is_file() else clean_data(load_raw_dataset())
    if df.empty:
        raise ValueError(f"O ficheiro CSV em {path} está vazio.")
    missing_cols = [c for c in TARGET_VARS + [TEAM_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")

    logger.info("Dados carregados de %s (%d linhas)", path, len(df))
    return df


def build_processed_dataset() -> Path:
    """Gera `data/processed/nba_cleaned.csv` a partir do CSV em `data/raw/`."""
    raw_file = _find_raw_csv()
    if raw_file is None:
        raise FileNotFoundError(f"Nenhum ficheiro CSV encontrado em {RAW_DIR}")

    raw = pd.read_csv(raw_file)
    print(f"[data_prep] Raw encontrado: {raw_file} ({len(raw)} linhas)")

    cleaned = clean_data(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"[data_prep] Processado gravado em {PROCESSED_PATH} ({len(cleaned)} linhas)")
    return PROCESSED_PATH


if __name__ == "__main__":
    build_processed_dataset()
