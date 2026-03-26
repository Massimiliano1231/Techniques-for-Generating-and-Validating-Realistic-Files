import os, csv
from pathlib import Path
from config.constants import EXTS
from data.datasets import DATASETS
from config.constants import METRIC_COL_TO_NAME
from tqdm import tqdm
from config.constants import MIN_REAL, K_SIGMA, RANGES
from core.bfd_features import ngram_bfd_from_path
from core.metrics import jsd, tvd, cosine_sim, entropy
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DETECTOR = PROJECT_ROOT / "data" / "detector"
DATA_GENERATOR = PROJECT_ROOT / "data" / "generator"

def list_files(folder, wanted_exts):
    out=[]
    if not os.path.isdir(folder): return out
    for dp,_,fns in os.walk(folder):
        for fn in fns:
            low=fn.lower()
            if any(low.endswith(e) for e in wanted_exts):
                out.append(os.path.join(dp, fn))
    return out

def load_thresholds(csv_path, fmt):
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("format")==fmt:
                def g(m,k): return float(row[f"{m}_{k}"])
                stats={}
                for m in ["JSD","TVD","L1","Cosine","Entropy"]:
                    stats[m]={"mean":g(m,"mean"), "std":g(m,"std"), "var":g(m,"var"), "p95":g(m,"p95")}
                return stats
    return None


def scan_your_layout(root):
    mapping = {
        "pdf": {
            "real": ["pdf data/PDF-total"],
            "rand": ["pdf data/pdf_ranflood"]
        },
        "txt": {
            "real": ["txt data/TXT-total"],
            "rand": ["txt data/txt_ranflood"]
        },
        "jpg": {
            "real": ["jpg data/JPG-total"],
            "rand": ["jpg data/jpg_ranflood"]
        },
        "docx": {
            "real": ["docx data/DOCX-total"],
            "rand": ["docx data/docx_ranflood"]
        }
    }

    real = {fmt: [] for fmt in mapping}
    rand = {fmt: [] for fmt in mapping}

    for fmt, d in mapping.items():
        # REAL
        for relpath in d["real"]:
            full = os.path.join(root, relpath)
            if os.path.isdir(full):
                real[fmt] += [os.path.join(full, x) for x in os.listdir(full)]

        # RANDOM
        for relpath in d["rand"]:
            full = os.path.join(root, relpath)
            if os.path.isdir(full):
                rand[fmt] += [os.path.join(full, x) for x in os.listdir(full)]

    return real, rand



import os

def scan_your_layout_gen(root):
    mapping = {
        "pdf": {
            "real": [
                str(DATA_GENERATOR / "generated_files" / "pdf")
            ],
            "rand": ["pdf data/pdf_ranflood"]
        },
        "txt": {
            "real": [
                str(DATA_GENERATOR / "generated_files" / "txt")
            ],
            "rand": ["txt data/txt_ranflood"]
        },
        "jpg": {
            "real": [
                str(DATA_GENERATOR / "generated_files" / "jpg")
            ],
            "rand": ["jpg data/jpg_ranflood"]
        },
        "docx": {
            "real": [
                str(DATA_GENERATOR / "generated_files" / "docx")
            ],
            "rand": ["docx data/docx_ranflood"]
        }
    }

    real = {fmt: [] for fmt in mapping}
    rand = {fmt: [] for fmt in mapping}

    for fmt, d in mapping.items():

        # -------- REAL (path assoluti) --------
        for path in d["real"]:
            full = path if os.path.isabs(path) else os.path.join(root, path)
            if os.path.isdir(full):
                real[fmt] += [
                    os.path.join(full, x)
                    for x in os.listdir(full)
                    if os.path.isfile(os.path.join(full, x))
                ]

        # -------- RANDOM (relativi a root) --------
        for path in d["rand"]:
            full = path if os.path.isabs(path) else os.path.join(root, path)
            if os.path.isdir(full):
                rand[fmt] += [
                    os.path.join(full, x)
                    for x in os.listdir(full)
                    if os.path.isfile(os.path.join(full, x))
                ]

    return real, rand







def load_optimized_thresholds(path_csv):
    """
    Legge thresholds_all_formats_train_fold{k}.csv e costruisce:
      thr_by_fmt[fmt][metric_name] = soglia (o (low,high) per Entropy)
    metric_name qui è uno tra: "JSD", "TVD", "L1",  "Cosine", "Entropy"
    """
    thr_by_fmt = {}
    with open(path_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fmt = row["format"].strip()
            metric_col = row["metric"].strip()
            if metric_col not in METRIC_COL_TO_NAME:
                continue
            mname = METRIC_COL_TO_NAME[metric_col]

            if fmt not in thr_by_fmt:
                thr_by_fmt[fmt] = {}

            if mname == "Entropy":
                if row["thr_low"] == "" or row["thr_high"] == "":
                    continue
                low = float(row["thr_low"])
                high = float(row["thr_high"])
                thr_by_fmt[fmt][mname] = (low, high)
            else:
                if row["threshold"] == "":
                    continue
                T = float(row["threshold"])
                thr_by_fmt[fmt][mname] = T

    return thr_by_fmt


def load_test_rows_by_format(scores_csv):
    """
    Legge file_scores_centroid_test_fold{k}.csv e raggruppa le righe per formato.
    Ritorna:
      rows_by_fmt[fmt] = [row1, row2, ...]
    """
    rows_by_fmt = {}
    with open(scores_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fmt = row["format"].strip()
            rows_by_fmt.setdefault(fmt, []).append(row)
    return rows_by_fmt



def apply_rules(vals, thr_fmt):
    """
    vals: dizionario con le metriche del file:
      {"JSD": ..., "TVD": ..., "L1": ...,: ..., "Cosine": ..., "Entropy": ...}

    thr_fmt: soglie per quel formato:
      {"JSD": T_jsd, ..., "Entropy": (low, high), ...}

    Ritorna:
      overall: True se tutte le metriche passano
      decisions: dict m -> True/False per singola metrica
    """
    decisions = {}
    for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]:
        if m not in thr_fmt:
            # se non abbiamo soglia per quella metrica → la consideriamo passata
            decisions[m] = True
            continue

        T = thr_fmt[m]

        if m == "Cosine":
            # alto è meglio
            decisions[m] = (vals[m] >= T)
        elif m == "Entropy" and isinstance(T, tuple):
            low, high = T
            decisions[m] = (low <= vals[m] <= high)
        else:
            # metriche di distanza: basso è meglio
            decisions[m] = (vals[m] <= T)

    overall = all(decisions.values())
    return overall, decisions





def process_format(fmt, thr_fmt, rows):
    """
    Calcola FN/FP sul TEST per un singolo formato,
    usando le soglie thr_fmt (ottenute dal TRAIN di quel fold)
    e le righe 'rows' (provenienti da file_scores_centroid_test_fold{k}.csv).
    """
    if not rows:
        print(f"[{fmt}] nessuna riga nel TEST, skip.")
        return None

    if not thr_fmt:
        print(f"[{fmt}] ATTENZIONE: nessuna soglia ottimizzata per questo formato, skip.")
        return None

    N_real = 0
    N_rand = 0
    FN = 0
    FP = 0

    metric_FN = {m: 0 for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]}
    metric_FP = {m: 0 for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]}

    for row in rows:
        cls = row["class"].strip().lower()
        try:
            vals = {
                "JSD":      float(row["jsd_mean"]),
                "TVD":      float(row["tvd_mean"]),
                "L1":       float(row["l1_mean"]),
                "Cosine":   float(row["cosine_sim_mean"]),
                "Entropy":  float(row["entropy"]),
            }
        except ValueError:
            # valori non parsabili → skip del file
            continue

        overall, decs = apply_rules(vals, thr_fmt)

        if cls == "real":
            N_real += 1
            if not overall:
                FN += 1
                for m, ok in decs.items():
                    if not ok:
                        metric_FN[m] += 1
        elif cls == "random":
            N_rand += 1
            if overall:
                FP += 1
            for m, ok in decs.items():
                if ok:
                    metric_FP[m] += 1

    if N_real < MIN_REAL:
        print(f"[{fmt}] ERRORE: real insufficienti nel TEST ({N_real})")
        return None

    # === CONFUSION MATRIX ===
    TP = N_real - FN
    TN = N_rand - FP

    confusion = {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN
    }

    res = {
        "fmt": fmt,
        "N_real": N_real,
        "N_rand": N_rand,
        "FN": FN,
        "FN_rate": (FN / N_real) if N_real else 0.0,
        "FP": FP,
        "FP_rate": (FP / N_rand) if N_rand else 0.0,
        "metric_FN": metric_FN,
        "metric_FP": metric_FP,
        "confusion_matrix": confusion
    }

    return res



def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load_sigma_thresholds(path_csv):
  
    thr_by_fmt = {}

    # mappa metrica -> prefisso di colonna nel CSV
    METRIC_TO_PREFIX = {
        "JSD": "JSD",
        "TVD": "TVD",
        "L1": "L1",
        "Cosine": "Cosine",
        "Entropy": "Entropy",
    }

    with open(path_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            fmt = row["format"].strip()
            thr_by_fmt[fmt] = {}

            for m, pref in METRIC_TO_PREFIX.items():
                mean_col = f"{pref}_mean"
                std_col  = f"{pref}_std"
                if mean_col not in row or std_col not in row:
                    continue
                try:
                    mu = float(row[mean_col])
                    sd = float(row[std_col])
                except ValueError:
                    continue

                thr_by_fmt[fmt][m] = {"mean": mu, "std": sd}

    return thr_by_fmt








def apply_rules_sigma(vals, thr_fmt):
    """
    vals: dict con le metriche del file:
      {"JSD": ..., "TVD": ..., "L1": ..., ..., "Cosine": ..., "Entropy": ...}

    thr_fmt: dict soglie "statistiche" per quel formato:
      {"JSD": {"mean": mu, "std": sd}, ..., "Entropy": {"mean": mu_e, "std": sd_e}}

    Usa le stesse regole dello script di plausibility:
      - Distanze: T = clamp(mu + 2σ, RANGES[m]), passa se val <= T
      - Cosine:   T = clamp(mu - 2σ, RANGES["Cosine"]), passa se val >= T
      - Entropy:  banda [mu - 2σ, mu + 2σ], con clamp ai limiti di RANGES["Entropy"]
    """
    decisions = {}

    # Distanze
    for m in ["JSD", "TVD", "L1"]:
        if m not in thr_fmt:
            decisions[m] = True
            continue
        mu = thr_fmt[m]["mean"]
        sd = thr_fmt[m]["std"]
        T = clamp(mu + K_SIGMA * sd, *RANGES[m])
        decisions[m] = (vals[m] <= T)

    # Cosine (similarità): lower bound = mu - 2σ
    if "Cosine" in thr_fmt:
        mu_c = thr_fmt["Cosine"]["mean"]
        sd_c = thr_fmt["Cosine"]["std"]
        T_cos = clamp(mu_c - K_SIGMA * sd_c, *RANGES["Cosine"])
        decisions["Cosine"] = (vals["Cosine"] >= T_cos)
    else:
        decisions["Cosine"] = True

    # Entropy: banda [mu - 2σ, mu + 2σ]
    if "Entropy" in thr_fmt:
        mu_e = thr_fmt["Entropy"]["mean"]
        sd_e = thr_fmt["Entropy"]["std"]
        E_low = clamp(mu_e - K_SIGMA * sd_e, *RANGES["Entropy"])
        E_high = clamp(mu_e + K_SIGMA * sd_e, *RANGES["Entropy"])
        decisions["Entropy"] = (E_low <= vals["Entropy"] <= E_high)
    else:
        decisions["Entropy"] = True

    overall = all(decisions.values())
    return overall, decisions


def process_format_sigma(fmt, thr_fmt, rows):
    """
    Calcola FN/FP usando le soglie μ±2σ per un singolo formato.
    """
    if not rows:
        print(f"[{fmt}] nessuna riga, skip.")
        return None

    if not thr_fmt:
        print(f"[{fmt}] ATTENZIONE: nessuna soglia (mean/std) per questo formato, skip.")
        return None

    N_real = 0
    N_rand = 0
    FN = 0
    FP = 0

    metric_FN = {m: 0 for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]}
    metric_FP = {m: 0 for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]}

    for row in rows:
        cls = row["class"].strip().lower()
        try:
            vals = {
                "JSD":      float(row["jsd_mean"]),
                "TVD":      float(row["tvd_mean"]),
                "L1":       float(row["l1_mean"]),
                "Cosine":   float(row["cosine_sim_mean"]),
                "Entropy":  float(row["entropy"]),
            }
        except ValueError:
            # valori non parsabili → skip
            continue

        overall, decs = apply_rules_sigma(vals, thr_fmt)

        if cls == "real":
            N_real += 1
            if not overall:
                FN += 1
                for m, ok in decs.items():
                    if not ok:
                        metric_FN[m] += 1
        elif cls == "random":
            N_rand += 1
            if overall:
                FP += 1
            for m, ok in decs.items():
                if ok:
                    metric_FP[m] += 1

    if N_real < MIN_REAL:
        print(f"[{fmt}] ERRORE: real insufficienti  ({N_real})")
        return None

    res = {
        "fmt": fmt,
        "N_real": N_real,
        "N_rand": N_rand,
        "FN": FN,
        "FN_rate": (FN / N_real) if N_real else 0.0,
        "FP": FP,
        "FP_rate": (FP / N_rand) if N_rand else 0.0,
        "metric_FN": metric_FN,
        "metric_FP": metric_FP,
    }
    return res




def compute_centroid_for_format(fmt, ngram, buckets):
    """
    Calcola la BFD media (centroide) sui REAL del formato.
    """
    print(f"\n--- {fmt.upper()} ---")

    real_dir = DATASETS[fmt]["real"]
    paths = list_files(real_dir, EXTS[fmt])

    if len(paths) < MIN_REAL:
        print(f"[ERRORE] Formato {fmt}: real insufficienti ({len(paths)})")
        return None

    bfds = []
    for p in tqdm(paths, desc=f"{fmt}: BFD real"):
        bfd = ngram_bfd_from_path(p, n=ngram, buckets=buckets)
        bfds.append(bfd)

    centroid = np.mean(bfds, axis=0)

    # normalizzazione (sommatoria = 1)
    s = centroid.sum()
    if s > 0:
        centroid = centroid / s

    return centroid.tolist()

def build_get_repr(ngram: int, buckets: int):
    def get_repr(path: str):
        return ngram_bfd_from_path(path, n=ngram, buckets=buckets)
    return get_repr


def compute_centroid(real_paths, get_repr):
    """
    Calcola il centroide (media vettoriale normalizzata) dei REAL_train.
    """
    if not real_paths:
        return None

    real_vecs = [get_repr(p) for p in real_paths]
    real_vecs = np.array(real_vecs, dtype=float)

    centroid = real_vecs.mean(axis=0)
    s = centroid.sum()
    if s > 0:
        centroid = centroid / s
    return centroid


def write_scores_for_group(fmt, real_paths, rand_paths, centroid, writer, get_repr, group_name: str, fold_idx: int):
    """
    Scrive una riga per ogni file (REAL + RANDOM) del gruppo (train o test)
    con tutte le metriche rispetto al centroide fornito.

    group_name: "TRAIN" o "TEST" (solo per log)
    fold_idx: indice del fold (0, 1, 2, ...)
    """
    if not real_paths and not rand_paths:
        print(f"[{fmt}][fold={fold_idx}][{group_name}] nessun file (REAL o RANDOM), skip.")
        return

    print(f"[{fmt}][fold={fold_idx}][{group_name}] REAL={len(real_paths)}, RANDOM={len(rand_paths)}")

    def write_row(path, cls):
        v = get_repr(path)
        j = jsd(v, centroid)
        t = tvd(v, centroid)
        l = float(np.sum(np.abs(v - centroid)))
        cos = cosine_sim(v, centroid)
        e = entropy(v)
        writer.writerow([
            fmt,
            path,
            cls,
            fold_idx,         # nuova colonna: indice del fold
            f"{j:.10f}",
            f"{t:.10f}",
            f"{l:.10f}",
            f"{cos:.10f}",
            f"{e:.10f}",
        ])

    for p in real_paths:
        write_row(p, "real")
    for p in rand_paths:
        write_row(p, "random")




def load_scores(input_csv, target_format, metric_name):
    scores_real = []
    scores_rand = []

    with open(input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"format", "class", metric_name}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"Nel CSV mancano colonne: {missing}")

        for row in reader:
            fmt = row["format"].strip()
            if fmt != target_format:
                continue

            cls = row["class"].strip().lower()
            try:
                val = float(row[metric_name])
            except ValueError:
                continue

            if cls == "real":
                scores_real.append(val)
            elif cls == "random":
                scores_rand.append(val)

    if not scores_real or not scores_rand:
        raise RuntimeError(
            f"Nessun dato sufficiente per format={target_format}, metric={metric_name} "
            f"(real={len(scores_real)}, random={len(scores_rand)})"
        )

    return np.array(scores_real), np.array(scores_rand)





def load_centroids(path_json):
    """
    centroids.json:
      { "pdf": [...], "txt": [...], "jpg": [...], "docx": [...] }

    Ritorna:
      centroids[fmt] = np.array(...)
    """
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    centroids = {}
    for fmt, vec in data.items():
        centroids[fmt] = np.array(vec, dtype=float)

    return centroids



def load_final_thresholds(path_csv):
    """
    Legge final_thresholds_mean.csv (output di compute_final_thresholds.py)
    e costruisce:

      thr_by_fmt[fmt]["JSD"]      = soglia (float)
      thr_by_fmt[fmt]["TVD"]      = soglia
      thr_by_fmt[fmt]["L1"]       = soglia
      thr_by_fmt[fmt]["Cosine"]   = soglia
      thr_by_fmt[fmt]["Entropy"]  = (low, high)

    metric nel CSV è uno tra:
      jsd_mean, tvd_mean, l1_mean, cosine_sim_mean, entropy
    """
    thr_by_fmt = {}

    with open(path_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fmt = row["format"].strip()
            metric_col = row["metric"].strip()

            if metric_col not in METRIC_COL_TO_NAME:
                continue

            mname = METRIC_COL_TO_NAME[metric_col]

            if fmt not in thr_by_fmt:
                thr_by_fmt[fmt] = {}

            # Entropy: banda [low, high]
            if mname == "Entropy":
                low_str = row.get("thr_low", "")
                high_str = row.get("thr_high", "")
                if low_str == "" or high_str == "":
                    continue
                try:
                    low = float(low_str)
                    high = float(high_str)
                except ValueError:
                    continue
                thr_by_fmt[fmt][mname] = (low, high)
            else:
                thr_str = row.get("threshold", "")
                if thr_str == "":
                    continue
                try:
                    T = float(thr_str)
                except ValueError:
                    continue
                thr_by_fmt[fmt][mname] = T

    return thr_by_fmt




def apply_rules_optimized(vals, thr_fmt):
    """
    vals: dict metriche del file:
      {"JSD": ..., "TVD": ..., "L1": ......, "Cosine": ..., "Entropy": ...}

    thr_fmt: soglie ottimizzate per quel formato:
      per distanze: T (float)
      per Cosine:   T_cos (float)
      per Entropy:  (low, high)

    Ritorna:
      overall: True se tutte le metriche passano
      decisions: dict m -> True/False
    """
    decisions = {}

    # distanze: basso è meglio → val <= T
    for m in ["JSD", "TVD", "L1"]:   
        if m not in thr_fmt:
            decisions[m] = True
            continue
        T = thr_fmt[m]
        decisions[m] = (vals[m] <= T)

    # Cosine: alto è meglio → val >= T_cos
    if "Cosine" in thr_fmt:
        T_cos = thr_fmt["Cosine"]
        decisions["Cosine"] = (vals["Cosine"] >= T_cos)
    else:
        decisions["Cosine"] = True

    # Entropy: banda [low, high]
    if "Entropy" in thr_fmt:
        low, high = thr_fmt["Entropy"]
        decisions["Entropy"] = (low <= vals["Entropy"] <= high)
    else:
        decisions["Entropy"] = True

    overall = all(decisions.values())
    return overall, decisions
