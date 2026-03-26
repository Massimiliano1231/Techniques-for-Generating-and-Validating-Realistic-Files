import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from data.datasets import DATASETS
from markov.bigram_counter import init_counter, update_bigram_counts
from markov.markov_builder import normalize_rows
from formats.extractor_bytes import read_structural_docx_bytes


OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def main():
    dataset_path = DATASETS["docx"]["real"]
    print(f"[+] Building DOCX Markov model from: {dataset_path}")

    docx_files = list(Path(dataset_path).rglob("*.docx"))
    print(f"[+] Found {len(docx_files)} DOCX files")

    C = init_counter()

    for path in tqdm(docx_files, desc="Processing DOCX files", unit="file"):
        try:
            data = read_structural_docx_bytes(path)
        except Exception:
            continue

        if not data or len(data) < 2:
            continue

        update_bigram_counts(C, data)

    print("[+] Normalizing counts to obtain transition matrix P_docx")
    P = normalize_rows(C)

    out_path = OUTPUT_DIR / "P_docx.npy"
    np.save(out_path, P)

    print(f"[+] Saved DOCX Markov matrix to {out_path}")
    print("[+] Done.")


if __name__ == "__main__":
    main()
