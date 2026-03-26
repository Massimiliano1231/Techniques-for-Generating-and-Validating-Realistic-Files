
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))


from config.constants import CSV_PATH, OUT_METRICHE, OUT_ENTROPIA, METRICS
from plotting.plot_utils import load_csv, ensure_dir, plot_metric_bars, plot_entropy_by_format






def main():
    print(f"[INFO] Leggo CSV: {CSV_PATH}")
    df = load_csv(CSV_PATH)

    ensure_dir(OUT_METRICHE)
    ensure_dir(OUT_ENTROPIA)

    for m in METRICS:
        plot_metric_bars(df, m, OUT_METRICHE)

    plot_entropy_by_format(df, OUT_ENTROPIA)

    print("[DONE] Grafici creati.")

if __name__ == "__main__":
    main()
