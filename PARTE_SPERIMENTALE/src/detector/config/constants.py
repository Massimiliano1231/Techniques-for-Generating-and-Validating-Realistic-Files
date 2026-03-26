NGRAM   = 2
BUCKETS = 65536

RANGES = {
    "JSD":      (0.0, 1.0),
    "TVD":      (0.0, 1.0),
    "L1":       (0.0, 2.0),
    "Cosine":   (-1.0, 1.0),
    "Entropy":  (0.0, 16.0),  
}

EXTS = {
    "pdf":  [".pdf"],
    "txt":  [".txt"],
    "jpg":  [".jpg", ".jpeg"],
    "docx": [".docx"],
}

METRIC_COL_TO_NAME = {
    "jsd_mean": "JSD",
    "tvd_mean": "TVD",
    "l1_mean": "L1",
    "cosine_sim_mean": "Cosine",
    "entropy": "Entropy",
}

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DETECTOR = PROJECT_ROOT / "data" / "detector"
RESULTS_DETECTOR = PROJECT_ROOT / "results" / "detector"

DEFAULT_THRESH_DIR = str(DATA_DETECTOR / "csv_utils" / "soglie_ottimizzate_per_ogni_fold")
DEFAULT_SCORES_DIR = str(DATA_DETECTOR / "csv_utils" / "csv_train_e_test_tutti_i_fold")


MIN_REAL = 2 

CSV_THRESHOLDS_OPT = str(DATA_DETECTOR / "csv_utils" / "soglie_ottimizzate_per_ogni_fold" / "thresholds_all_formats_train.csv")
CSV_SCORES_TEST = str(DATA_DETECTOR / "csv_utils" / "csv_train_e_test_un_fold" / "file_scores_centroid_test.csv")


K_SIGMA = 3.0
DEFAULT_VAR_CSV = str(DATA_DETECTOR / "derived" / "vari_csv" / "csv_varianza" / "variance_from_mean_summary.csv")
DEFAULT_SCORES_CSV = str(DATA_DETECTOR / "csv_utils" / "csv_train_e_test_un_fold" / "file_scores_centroid_train.csv")


CSV_PATH = str(DATA_DETECTOR / "derived" / "vari_csv" / "csv_distanza_coppie_random_vs_real" / "pairwise_random_vs_real.csv")
OUT_METRICHE = str(RESULTS_DETECTOR / "grafici" / "distanza_real_random" / "metriche_normali")
OUT_ENTROPIA = str(RESULTS_DETECTOR / "grafici" / "distanza_real_random" / "metrica_entropia")


METRICS = ["jsd", "tvd", "l1", "cosine_sim"]



metricsForStampa = ["jsd", "tvd", "l1","cosine_sim", "entropy_real", "entropy_generated"]

metricsFForStampaGen = ["jsd", "tvd", "l1","cosine_sim", "entropy"]

OUT_JSON = str(DATA_DETECTOR / "derived" / "vari_json" / "centroidi_ogni_formato" / "centroids.json")



DEFAULT_KFOLD_JSON = str(DATA_DETECTOR / "derived" / "vari_json" / "json_split_dataset" / "kfold_split_3.json")

DEFAULT_SPLIT_JSON = str(DATA_DETECTOR / "csv_utils" / "json_split_dataset_per_un_fold" / "train_test_split.json")




DEFAULT_CENTROIDS_JSON = str(DATA_DETECTOR / "derived" / "vari_json" / "centroidi_ogni_formato" / "centroids.json")
DEFAULT_THRESHOLDS_CSV = str(DATA_DETECTOR / "derived" / "vari_csv" / "csv_soglie_finali" / "final_thresholds_mean.csv")


DEFAULT_ROOT = str(DATA_DETECTOR / "datasets")
DEFAULT_CSV = str(DATA_DETECTOR / "derived" / "vari_csv" / "csv_varianza" / "variance_from_mean_summary.csv")
