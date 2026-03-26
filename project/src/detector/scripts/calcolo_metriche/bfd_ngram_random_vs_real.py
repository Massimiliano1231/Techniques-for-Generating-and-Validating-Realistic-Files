"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))

import os, argparse, random, csv
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cityblock

from core.metrics import jsd, tvd, cosine_sim, entropy
from io.io_utils import scan_your_layout, build_get_repr, scan_your_layout_gen






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Es: data/detector/datasets")
    ap.add_argument("--out", required=True, help="Cartella output")
    ap.add_argument("--pairs", type=int, default=20000, help="Coppie random-vs-real per formato")
    ap.add_argument("--ngram", type=int, default=2, help="Ordine n-gram (1=BFD, 2=bigram, 3=trigram, ...)")
    ap.add_argument("--buckets", type=int, default=4096, help="Numero di bucket per hashing n-gram (per n=2, 65536 = esatto)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    real_by_fmt, rand_by_fmt = scan_your_layout_gen(args.root)
    for fmt in ["pdf","txt","jpg","docx"]:
        print(fmt,
              "REAL:", len(real_by_fmt[fmt]),
              "RAND:", len(rand_by_fmt[fmt]))

   
    out_csv = os.path.join(args.out, "pairwise_random_vs_real.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["format","file_random","file_real","jsd","tvd","l1",
                    "cosine_sim","entropy_random","entropy_real"])

        for fmt in ["pdf","txt","jpg","docx"]:
            R, L = rand_by_fmt[fmt], real_by_fmt[fmt]
            if not R or not L:
                print(f"[{fmt}] skip: mancano file (random={len(R)}, real={len(L)})")
                continue
            n_pairs = min(args.pairs, len(R)*len(L))
            desc = f"{fmt} random-vs-real (n={args.ngram}, buckets={args.buckets})"
            get_repr = build_get_repr(args.ngram, args.buckets)
            for _ in tqdm(range(n_pairs), desc=desc):
                A, B = random.choice(R), random.choice(L)
                p = get_repr(A)
                q = get_repr(B)
                w.writerow([
                    fmt, A, B,
                    jsd(p,q),
                    tvd(p,q),
                    float(cityblock(p,q)),
                    cosine_sim(p,q),
                    entropy(p),
                    entropy(q)
                ])

    print("OK ->", out_csv)

if __name__ == "__main__":
    main()
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))

import os, argparse, random, csv
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cityblock

from core.metrics import jsd, tvd, cosine_sim, entropy
from io.io_utils import build_get_repr, scan_your_layout, scan_your_layout_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root dei dataset REAL (DETECTOR)")
    ap.add_argument("--out", required=True,
                    help="Cartella output")
    ap.add_argument("--pairs", type=int, default=20000,
                    help="Coppie real-vs-generati per formato")
    ap.add_argument("--ngram", type=int, default=2,
                    help="Ordine n-gram (1=BFD, 2=bigram, 3=trigram, ...)")
    ap.add_argument("--buckets", type=int, default=4096,
                    help="Numero di bucket per hashing n-gram")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -------- REAL (dataset veri, sotto root) --------
    real_by_fmt, _ = scan_your_layout(args.root)

    # -------- GENERATI (file sintetici plausibili) --------
    gen_by_fmt, _ = scan_your_layout_gen(args.root)

    for fmt in ["pdf", "txt", "jpg", "docx"]:
        print(fmt,
              "REAL:", len(real_by_fmt[fmt]),
              "GENERATED:", len(gen_by_fmt[fmt]))

    out_csv = os.path.join(args.out, "pairwise_real_vs_generated.csv")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "format",
            "file_real",
            "file_generated",
            "jsd",
            "tvd",
            "l1",
            "cosine_sim",
            "entropy_real",
            "entropy_generated"
        ])

        for fmt in ["pdf", "txt", "jpg", "docx"]:
            R = real_by_fmt[fmt]
            G = gen_by_fmt[fmt]

            if not R or not G:
                print(f"[{fmt}] skip: mancano file (real={len(R)}, gen={len(G)})")
                continue

            n_pairs = min(args.pairs, len(R) * len(G))
            desc = f"{fmt} real-vs-generated (n={args.ngram}, buckets={args.buckets})"

            get_repr = build_get_repr(args.ngram, args.buckets)

            for _ in tqdm(range(n_pairs), desc=desc):
                A = random.choice(R)   # REAL
                B = random.choice(G)   # GENERATO

                p = get_repr(A)
                q = get_repr(B)

                w.writerow([
                    fmt,
                    A,
                    B,
                    jsd(p, q),
                    tvd(p, q),
                    float(cityblock(p, q)),
                    cosine_sim(p, q),
                    entropy(p),
                    entropy(q)
                ])

    print("OK ->", out_csv)


if __name__ == "__main__":
    main()
