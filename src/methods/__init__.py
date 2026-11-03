"""
Pacote `src.methods` — implementações individuais de cada método estatístico
e algoritmo de deteção (um ficheiro por método).

Métodos de Diagnóstico de Dados Ausentes (Missing):
    * little_test.py          -> Teste MCAR de Little (1988)
    * jamshidian_jalal.py     -> Teste de Jamshidian-Jalal (2010)
    * listing_schlittgen.py   -> Teste de Listing e Schlittgen (LS, 1998)
    * fairclough_logistic.py  -> Regressão Logística de Fairclough (1998/2002)

Métodos de Deteção de Outliers / Anomalias:
    * iqr_boxplot.py          -> IQR (Intervalo Interquartil) / Boxplot e extremos
    * zscore.py               -> Z-Score Clássico e Z-Score Modificado (MAD)
    * grubbs.py               -> Teste de Grubbs (ESD, 1950)
    * ecod.py                 -> ECOD (Empirical Cumulative Distribution-based Outlier Detection)
    * knn.py                  -> K-NN (K-Nearest Neighbors) para anomalias
    * dbscan.py               -> DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    * lof.py                  -> LOF (Local Outlier Factor)
"""

from .dbscan import DBSCAN_THEORY, detect_outliers_dbscan
from .ecod import ECOD_THEORY, detect_outliers_ecod
from .fairclough_logistic import FAIRCLOUGH_LOGISTIC_GUIDE, fairclough_logistic_test
from .grubbs import GRUBBS_THEORY, detect_outliers_grubbs, grubbs_test_summary
from .iqr_boxplot import IQR_THEORY, detect_outliers_iqr, iqr_bounds
from .jamshidian_jalal import JAMSHIDIAN_TEST_GUIDE, jamshidian_jalal_test
from .knn import KNN_THEORY, detect_outliers_knn
from .listing_schlittgen import LISTING_SCHLITTGEN_GUIDE, listing_schlittgen_test
from .little_test import LITTLE_TEST_GUIDE, littles_mcar_test
from .lof import LOF_THEORY, detect_outliers_lof
from .zscore import ZSCORE_THEORY, detect_outliers_modified_zscore, detect_outliers_zscore

__all__ = [
    "LITTLE_TEST_GUIDE",
    "littles_mcar_test",
    "JAMSHIDIAN_TEST_GUIDE",
    "jamshidian_jalal_test",
    "LISTING_SCHLITTGEN_GUIDE",
    "listing_schlittgen_test",
    "FAIRCLOUGH_LOGISTIC_GUIDE",
    "fairclough_logistic_test",
    "IQR_THEORY",
    "iqr_bounds",
    "detect_outliers_iqr",
    "ZSCORE_THEORY",
    "detect_outliers_zscore",
    "detect_outliers_modified_zscore",
    "GRUBBS_THEORY",
    "grubbs_test_summary",
    "detect_outliers_grubbs",
    "ECOD_THEORY",
    "detect_outliers_ecod",
    "KNN_THEORY",
    "detect_outliers_knn",
    "DBSCAN_THEORY",
    "detect_outliers_dbscan",
    "LOF_THEORY",
    "detect_outliers_lof",
]

