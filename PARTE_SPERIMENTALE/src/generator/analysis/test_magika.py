import argparse
from pathlib import Path
from collections import Counter
import magika  
import time

EXTENSIONS = {
    "docx": [".docx"],
    "pdf": [".pdf"],
    "txt": [".txt"],
    "jpg": [".jpg"],
}

def unify_magika_label(desc: str) -> str | None:
    d = desc.lower()

    if "unknown binary data" in d:
        return "unknown"

    if "pdf" in d:
        return "pdf"
    if "jpeg" in d or "jpg" in d:
        return "jpg"
    if "word" in d or "docx" in d:
        return "docx"
    if "text" in d or "csv" in d:
        return "txt"

    return None  # altri formati

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory da analizzare"
    )
    args = parser.parse_args()

    files = sorted(f for f in args.input_dir.iterdir() if f.is_file())
    total = len(files)

    count_ext = Counter()
    count_magika = Counter()
    count_unknown = 0
    count_other = 0

    m = magika.Magika()

    start_time = time.perf_counter()

    for f in files:
        # -------- estensioni --------
        ext = f.suffix.lower()
        for fmt, exts in EXTENSIONS.items():
            if ext in exts:
                count_ext[fmt] += 1
                break

        # -------- Magika --------
        result = m.identify_path(f)

        if not result.ok:
            continue  # caso raro, ignoriamo

        label = unify_magika_label(result.output.description)

        if label == "unknown":
            count_unknown += 1
        elif label in {"docx", "pdf", "jpg", "txt"}:
            count_magika[label] += 1
        else:
            count_other += 1

    # =========================
    # STAMPA (come la vuoi tu)
    # =========================

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"\nTempo totale di analisi Magika: {elapsed:.2f} secondi")
    print(f"Tempo medio per file: {(elapsed / total if total else 0):.6f} s/file")

    print(f"FILE TOT: {total}")

    for fmt in ["docx", "pdf", "txt", "jpg"]:
        print(f"{fmt.upper()} EXTENSION: {count_ext[fmt]}")

        if fmt == "txt":
            print(f"MAGIKA TXT/CSV: {count_magika['txt']}")
        else:
            ext_cnt = count_ext[fmt]
            mag_cnt = count_magika[fmt]
            pct = mag_cnt / ext_cnt * 100 if ext_cnt > 0 else 0
            print(f"{fmt.upper()} MAGIKA: {mag_cnt} ({pct:.2f}%)")

    unknown_pct = count_unknown / total * 100 if total > 0 else 0
    other_pct = count_other / total * 100 if total > 0 else 0

    print(f"\nMAGIKA UNKNOWN: {count_unknown} ({unknown_pct:.2f}%)")
    print(f"MAGIKA ALTRI FORMATI: {count_other} ({other_pct:.2f}%)")

if __name__ == "__main__":
    main()
