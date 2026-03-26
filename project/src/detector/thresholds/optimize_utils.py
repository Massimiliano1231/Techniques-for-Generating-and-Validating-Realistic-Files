from scipy.optimize import dual_annealing
import os
import numpy as np
from io.io_utils import load_scores
from thresholds.objective import make_objective_default, make_objective_cosine, make_objective_entropy_band
import csv
import json



def optimize_for_fold(fold_idx, input_csv, alpha, beta, out_csv, json_dir=""):
    """
    Esegue l'ottimizzazione soglie per TUTTI i formati e TUTTE le metriche
    su un singolo CSV di TRAIN (di un fold).
    """
    formats = ["pdf", "txt", "jpg", "docx"]
    metrics = ["jsd_mean", "tvd_mean", "l1_mean",
               "cosine_sim_mean", "entropy"]

    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    results = []

    print(f"\n=== FOLD {fold_idx}: ottimizzazione soglie su {input_csv} ===")

    for fmt in formats:
        for metric_name in metrics:
            print(f"\n--- Fold={fold_idx}, format={fmt}, metric={metric_name} ---")
            try:
                scores_real, scores_rand = load_scores(input_csv, fmt, metric_name)
            except RuntimeError as e:
                print(f"SKIP: {e}")
                continue

            print(f"Caricati {len(scores_real)} REAL e {len(scores_rand)} RANDOM")

            all_vals = np.concatenate([scores_real, scores_rand])
            lower = float(all_vals.min())
            upper = float(all_vals.max())
            print(f"Range valori osservati: [{lower:.6f}, {upper:.6f}]")

            is_entropy = (metric_name == "entropy")
            is_cosine = (metric_name == "cosine_sim_mean")

            if is_entropy:
                objective, fn_fp = make_objective_entropy_band(
                    scores_real, scores_rand, alpha, beta
                )
                bounds = [(lower, upper), (lower, upper)]  # low, high
                res = dual_annealing(objective, bounds=bounds, maxiter=200)
                fn_best, fp_best, low_best, high_best = fn_fp(res.x)
                best_threshold = None
                thr_low = float(min(low_best, high_best))
                thr_high = float(max(low_best, high_best))
            else:
                if is_cosine:
                    objective, fn_fp_rates = make_objective_cosine(
                        scores_real, scores_rand, alpha, beta
                    )
                else:
                    objective, fn_fp_rates = make_objective_default(
                        scores_real, scores_rand, alpha, beta
                    )

                bounds = [(lower, upper)]
                res = dual_annealing(objective, bounds=bounds, maxiter=200)
                best_threshold = float(res.x[0])
                fn_best, fp_best = fn_fp_rates(best_threshold)
                thr_low = ""
                thr_high = ""

            best_cost = float(res.fun)

            print(f"  -> costo: {best_cost:.10f}")
            print(f"  -> FN_rate: {fn_best*100:.3f}%")
            print(f"  -> FP_rate: {fp_best*100:.3f}%")
            if is_entropy:
                print(f"  -> banda entropy: [{thr_low:.6f}, {thr_high:.6f}]")
            else:
                print(f"  -> soglia: {best_threshold:.10f}")

            row = {
                "fold": fold_idx,
                "format": fmt,
                "metric": metric_name,
                "alpha": alpha,
                "beta": beta,
                "threshold": best_threshold if best_threshold is not None else "",
                "thr_low": thr_low,
                "thr_high": thr_high,
                "cost": best_cost,
                "FN_rate": fn_best,
                "FP_rate": fp_best,
                "min_score": lower,
                "max_score": upper,
                "n_real": int(len(scores_real)),
                "n_random": int(len(scores_rand)),
            }
            results.append(row)

            if json_dir:
                json_path = os.path.join(
                    json_dir,
                    f"threshold_fold{fold_idx}_{fmt}_{metric_name}.json"
                )
                with open(json_path, "w") as jf:
                    json.dump(row, jf, indent=2)
                print(f"  -> JSON salvato in: {json_path}")

    # Scrivi CSV per questo fold
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "fold", "format", "metric", "alpha", "beta",
        "threshold", "thr_low", "thr_high", "cost",
        "FN_rate", "FP_rate",
        "min_score", "max_score", "n_real", "n_random"
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n=== DONE FOLD {fold_idx} ===")
    print(f"Soglie ottimizzate (fold {fold_idx}) salvate in: {out_csv}")
