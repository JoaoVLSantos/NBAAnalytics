# 🏀 NBA Analytics — Qualidade de Dados, Outliers e Estatística Descritiva

🌐 **Aplicação Online (Vercel):** [https://nba-analytics-sooty.vercel.app/](https://nba-analytics-sooty.vercel.app/)

Aplicação web interativa desenvolvida em **Flask** e **Plotly** para auditoria de qualidade de dados, diagnóstico formal de valores ausentes (*missing values*), detecção univariada e multivariada de *outliers* e análise estatística descritiva sobre estatísticas oficiais de partidas da temporada regular da **NBA temporada 25/26** (`2 238` registos e `29` variáveis).

---

## 📂 Estrutura do Projeto

```text
NBAAnalytics/
├── data/
│   ├── raw/                             # Base bruta original (stats_times_regular_final - stats_times_regular_final.csv)
│   └── processed/                       # Base limpa e padronizada (nba_cleaned.csv)
├── src/                                 # Camada analítica e estatística (desacoplada do Flask)
│   ├── methods/                         # Implementação modular dos testes e algoritmos
│   │   ├── little_test.py               # Teste Global MCAR de Little (1988)
│   │   ├── jamshidian_jalal.py          # Teste Não-Paramétrico de Jamshidian & Jalal (2010)
│   │   ├── listing_schlittgen.py        # Teste de Listing & Schlittgen (1998)
│   │   ├── fairclough_logistic.py       # Regressão Logística de Fairclough (2002)
│   │   ├── iqr_boxplot.py               # Cercas de Tukey / Boxplot (1,5×IQR e 3,0×IQR)
│   │   ├── zscore.py                    # Z-Score Clássico (3σ) e Z-Score Modificado (MAD)
│   │   ├── grubbs.py                    # Teste Paramétrico de Grubbs (ESD)
│   │   ├── ecod.py                      # ECOD (Empirical Cumulative Distribution Outlier Detection)
│   │   ├── knn.py                       # K-NN (K-Nearest Neighbors) para anomalias
│   │   ├── dbscan.py                    # DBSCAN (Agrupamento Espacial Baseado em Densidade)
│   │   └── lof.py                       # LOF (Local Outlier Factor)
│   ├── data_prep.py                     # Limpeza, auditoria de escala (÷1000), extração de siglas e metadados
│   ├── missing_analysis.py              # Orquestração de diagnóstico, simulação e imputação de missing
│   ├── outlier_detection.py             # Orquestração da auditoria por coluna e métodos de outliers
│   └── stats_metrics.py                 # Cálculo de medidas de tendência central, dispersão, separatrizes e forma
├── static/
│   ├── css/style.css                    # Tema escuro (Glassmorphism) e indicador flutuante de rolagem
│   └── js/charts.js                     # Renderização responsiva dos gráficos Plotly e controle de UI
├── templates/                           # Templates Jinja2 + Bootstrap 5
│   ├── base.html                        # Layout base, barra de navegação e seta indicadora de conteúdo
│   ├── index.html                       # Página inicial: visão geral da base, KPIs e dicionário de dados
│   ├── data_cleaning.html               # Análise técnica de qualidade, missing, testes formais e imputação
│   ├── outliers.html                    # Análise técnica de inconsistências e detecção de outliers
│   ├── descriptive.html                 # Estatística descritiva univariada por franquia e variável
│   └── theory.html                      # Fundamentação teórica completa com atalhos para aplicação prática
├── tests/                               # Suíte de testes automatizados (20 testes unitários e de integração)
│   ├── __init__.py                      # Inicializador da suíte de testes
│   ├── test_unit_data_prep.py           # Testes unitários de limpeza, extração de siglas e auditoria de escala
│   ├── test_unit_missing_analysis.py    # Testes unitários de missing, imputação e dos 4 testes formais
│   ├── test_unit_outlier_detection.py   # Testes unitários dos 7 métodos de detecção de outliers e inconsistências
│   ├── test_unit_stats_metrics.py       # Testes unitários das medidas descritivas e interpretações
│   └── test_integration_routes.py       # Testes de integração end-to-end das 5 rotas Flask (GET/POST/404)
├── app.py                               # Aplicação Flask (rotas HTTP e construção das figuras Plotly)
├── vercel.json                          # Configuração de deploy Serverless na Vercel (@vercel/python)
├── requirements.txt                     # Dependências de produção
├── .gitignore                           # Exclusões do Git
└── .vercelignore                        # Exclusões de deploy da Vercel
```

---

## 🧭 Funcionalidades e Rotas da Aplicação

| Rota | Aba na Interface | Descrição |
| :--- | :--- | :--- |
| `/` | **Visão Geral** | Resumo executivo da base da NBA (`2 238` jogos, `30` franquias, `29` variáveis), amostra dos registos e **Dicionário de Metadados** completo (tipo estatístico, descrição e unidade). |
| `/qualidade` | **Qualidade & Missing** | Levantamento da quantidade de campos com valores ausentes nas `29` colunas, aplicação dos **4 testes formais de mecanismo de missing** (na base real ou em cenários simulados MCAR/MAR), comparador interativo de **imputação** (`Mediana`, `Média`, `Zero`) e **auditoria de escala** do CSV bruto. |
| `/outliers` | **Detecção de Outliers** | Levantamento por coluna da quantidade de **valores inconsistentes** e **outliers**, filtro interativo por franquia/variável/método e comparação entre **9 abordagens** (univariadas e multivariadas) com gráficos Boxplot e Dispersão Bivariada (`MIN` vs Variável). |
| `/descritiva` | **Estatística Descritiva** | Cálculo detalhado por equipa e métrica de medidas de **Tendência Central** (Média, Mediana, Moda), **Dispersão** (Variância amostral, Desvio Padrão, Amplitude, CV%), **Separatrizes** ($Q_1$, $Q_2$, $Q_3$, IQR, Mínimo, Máximo) e **Forma** (Assimetria de Fisher-Pearson ajustada $G_1$), acompanhado de Histograma + Boxplot marginal. |
| `/teoria` | **Explicações e Base Teórica** | Guia teórico dedicado que explica **o que são**, **como funcionam (fórmulas)** e **por que utilizar** os mecanismos **MCAR, MAR e MNAR**, os **4 testes de missing** e os **7 grupos de métodos de outliers**, com botões diretos para executar cada método nos dados. |

---

## 🔬 Metodologia Analítica Implementada

### 1. Limpeza, Inconsistências e Padronização ([`src/data_prep.py`](src/data_prep.py))
* **Auditoria de Escala (`÷ 1 000`)**: Identifica e corrige automaticamente registos do CSV bruto em que proporções (`W_PCT`, `FG_PCT`, `FG3_PCT`, `FT_PCT`) foram exportadas multiplicadas por `1 000` (`> 1.0`) devido à perda do separador decimal.
* **Extração de Siglas e Remoção de Duplicados**: Padroniza nomes de colunas, remove linhas duplicadas, higieniza textos e extrai `TEAM_ABBREVIATION` a partir de `MATCHUP` (ou gera sigla de 3 letras a partir de `TEAM_NAME`).
* **Validação de Domínio e Fallback Serverless**: Converte contagens negativas inválidas em `NaN` e suporta processamento em memória para sistemas de arquivos *read-only* (como a Vercel).

### 2. Mecanismos e Testes Estatísticos de Valores Ausentes ([`src/missing_analysis.py`](src/missing_analysis.py))
* **Mecanismos Clássicos (Rubin, 1976)**:
  * **MCAR (*Missing Completely at Random*)**: A ausência independe de valores observados ou não observados.
  * **MAR (*Missing at Random*)**: A ausência depende de outras variáveis observadas na base (ex.: métricas condicionais a `REB` e `AST`).
  * **MNAR (*Missing Not at Random*)**: A ausência depende do próprio valor não observado.
* **Testes Formais Implementados**:
  1. **Teste de Little (1988)** ([`src/methods/little_test.py`](src/methods/little_test.py)): Estatística global $d^2 = \sum n_g (\bar{y}_{obs,g} - \hat{\mu}_{obs,g})^T \hat{\Sigma}_{obs,g}^{-1} (\bar{y}_{obs,g} - \hat{\mu}_{obs,g}) \sim \chi^2_{df}$.
  2. **Teste de Jamshidian & Jalal (2010)** ([`src/methods/jamshidian_jalal.py`](src/methods/jamshidian_jalal.py)): Avaliação de homocedasticidade/normalidade e teste não-paramétrico de $k$-amostras de **Anderson-Darling** entre padrões de ausência.
  3. **Teste de Listing & Schlittgen (1998)** ([`src/methods/listing_schlittgen.py`](src/methods/listing_schlittgen.py)): Comparação das médias das variáveis completamente observadas entre o grupo com ausência ($R=1$) e o grupo observado ($R=0$).
  4. **Regressão Logística de Fairclough (2002)** ([`src/methods/fairclough_logistic.py`](src/methods/fairclough_logistic.py)): Modelo $\text{logit}[P(R_Y = 1 \mid X)] = \beta_0 + \sum \beta_j X_j$ com teste de Razão de Verossimilhança ($G^2$) para distinguir MCAR vs. MAR.

### 3. Métodos de Detecção de Outliers ([`src/outlier_detection.py`](src/outlier_detection.py))
* **Univariados**:
  * **IQR / Boxplot (Cercas de Tukey)** ([`src/methods/iqr_boxplot.py`](src/methods/iqr_boxplot.py)): Limites moderados $[Q_1 - 1{,}5 \cdot \text{IQR},\; Q_3 + 1{,}5 \cdot \text{IQR}]$ e extremos $[Q_1 - 3{,}0 \cdot \text{IQR},\; Q_3 + 3{,}0 \cdot \text{IQR}]$.
  * **Z-Score Clássico ($3\sigma$) e Modificado (MAD)** ([`src/methods/zscore.py`](src/methods/zscore.py)): $|z| = |x - \bar{x}|/s > 3{,}0$ e $|M_i| = 0{,}6745 \cdot |x_i - \tilde{x}| / \text{MAD} > 3{,}5$ (Iglewicz & Hoaglin).
  * **Teste de Grubbs (ESD)** ([`src/methods/grubbs.py`](src/methods/grubbs.py)): $G = \max |x_i - \bar{x}| / s$ comparado ao valor crítico $t_{\alpha/(2N),\; N-2}$.
  * **ECOD Univariado (Li et al., 2022)** ([`src/methods/ecod.py`](src/methods/ecod.py)): Pontuação não-paramétrica livre de hiperparâmetros baseada nas caudas da ECDF empírica.
* **Multivariados (`PTS`, `REB`, `AST`, `STL`, `BLK`, `PF`, `FG3M`)**:
  * **ECOD Multivariado** ([`src/methods/ecod.py`](src/methods/ecod.py)): Agregação das log-probabilidades de cauda conjunta nas 7 dimensões.
  * **K-NN (Ramaswamy et al., 2000)** ([`src/methods/knn.py`](src/methods/knn.py)): Distância euclidiana padronizada ao $k$-ésimo vizinho mais próximo ($k=5$).
  * **DBSCAN (Ester et al., 1996)** ([`src/methods/dbscan.py`](src/methods/dbscan.py)): Agrupamento por densidade ($\varepsilon = 1{,}6$, $\text{minPts} = 5$), rotulando pontos de ruído (`label = -1`) como anomalias.
  * **LOF (Breunig et al., 2000)** ([`src/methods/lof.py`](src/methods/lof.py)): Rácio da densidade local de alcançabilidade face aos $k=20$ vizinhos mais próximos.

### 4. Estatística Descritiva Univariada ([`src/stats_metrics.py`](src/stats_metrics.py))
* **Variância e Desvio Padrão Amostrais** (`ddof=1`, divisor $n - 1$).
* **Quartis por Interpolação Linear** (método padrão do `pandas` / *type 7* de Hyndman & Fan).
* **Assimetria de Fisher-Pearson Ajustada ($G_1$)** (`scipy.stats.skew(bias=False)`) e classificação qualitativa (Bulmer, 1979).
* **Coeficiente de Variação ($\text{CV\%}$)**: $(s / \bar{x}) \times 100$ com classificação de dispersão relativa.

---

## 🧪 Suíte de Testes Unitários e de Integração (`tests/`)

O projeto conta com **20 testes automatizados** divididos em 5 módulos dentro da pasta [`tests/`](tests/):

| Arquivo de Teste | Tipo | Escopo Validado |
| :--- | :--- | :--- |
| [`tests/test_unit_data_prep.py`](tests/test_unit_data_prep.py) | **Unitário** | Remoção de duplicados, extração de `TEAM_ABBREVIATION` a partir de `MATCHUP`, correção automática de escala (`÷ 1 000`) em colunas de proporção (`W_PCT`, `FG_PCT`, `FG3_PCT`, `FT_PCT`), substituição de valores negativos por `NaN` e carregamento das `2 238` partidas reais. |
| [`tests/test_unit_missing_analysis.py`](tests/test_unit_missing_analysis.py) | **Unitário** | Resumo de valores ausentes por coluna e por franquia, imputação (`median`, `mean`, `zero` e erro em estratégia inválida) e execução dos **4 testes formais** (*Little*, *Jamshidian-Jalal*, *Listing & Schlittgen* e *Fairclough*) nos modos `real`, `mcar` e `mar`. |
| [`tests/test_unit_outlier_detection.py`](tests/test_unit_outlier_detection.py) | **Unitário** | Validação matemática de **IQR** ($1{,}5\times\text{IQR}$ e $3{,}0\times\text{IQR}$), **Z-Score ($3\sigma$)**, **Z-Score Modificado (MAD)**, **Teste de Grubbs**, **ECOD**, **K-NN**, **DBSCAN**, **LOF** e auditoria de inconsistências e outliers por coluna. |
| [`tests/test_unit_stats_metrics.py`](tests/test_unit_stats_metrics.py) | **Unitário** | Cálculo das medidas de tendência central, separatrizes, dispersão amostral e assimetria (incluindo casos limite $n=0$ e $n=1$) e funções de interpretação qualitativa. |
| [`tests/test_integration_routes.py`](tests/test_integration_routes.py) | **Integração** | Testes *end-to-end* com `app.test_client()` nas 5 rotas (`/`, `/qualidade`, `/outliers`, `/descritiva`, `/teoria`), modos de simulação, todos os 9 métodos de outliers, requisições `GET`/`POST`, *fallback* defensivo contra parâmetros inválidos e rota `404`. |

### Como executar os testes:
```powershell
python -m unittest discover -s tests -v
```

---

## 💻 Instalação e Execução Local

### 1. Clonar o repositório
```bash
git clone https://github.com/JoaoVLSantos/NBAAnalytics.git
cd NBAAnalytics
```

### 2. Criar e ativar o ambiente virtual
```bash
python -m venv venv
```

* **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (CMD)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
* **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar as dependências
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar a aplicação
```bash
python app.py
```
Aceda a **http://127.0.0.1:5000** no navegador.

---

## ☁️ Hospedagem na Vercel

A aplicação encontra-se publicada em produção na **Vercel**:
* **URL de Produção:** [https://nba-analytics-sooty.vercel.app/](https://nba-analytics-sooty.vercel.app/)

O projeto está configurado para execução **Serverless** através do arquivo [`vercel.json`](vercel.json). Qualquer novo `git push` para a branch `main` no GitHub atualiza automaticamente o deploy na Vercel.
