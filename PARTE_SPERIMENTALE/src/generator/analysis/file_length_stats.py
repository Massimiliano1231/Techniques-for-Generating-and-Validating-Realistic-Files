import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def main():
   
    project_root = Path(__file__).resolve().parents[3]
    input_dir = project_root / "data" / "generator" / "generated_files" / "all_generated"

    lengths = defaultdict(list)

    for file in input_dir.iterdir():
        if not file.is_file():
            continue

        ext = file.suffix.lower().lstrip(".")
        if ext not in {"pdf", "docx", "jpg", "txt"}:
            continue

        size = file.stat().st_size
        lengths[ext].append(size)

    print("\n[+] File length statistics (bytes)\n")

    for fmt, vals in sorted(lengths.items()):
        vals = np.array(vals)

        print(
            f"{fmt.upper():4s} | "
            f"N = {len(vals):5d} | "
            f"mean = {vals.mean():8.1f} | "
            f"std = {vals.std():8.1f} | "
            f"var = {vals.var():10.1f} | "
            f"min = {vals.min():6d} | "
            f"max = {vals.max():6d}"
        )

if __name__ == "__main__":
    main()
