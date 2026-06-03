import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from generator.formats.jpg_helper import (
    iter_jpegs, read_bytes, parse_jpeg_segments,
    build_bigram_markov, build_byte_markov
)

DATASET_REAL = PROJECT_ROOT / "data" / "detector" / "datasets" / "jpg data" / "JPG-total"
OUT_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"


def parse_args():
    parser = argparse.ArgumentParser(description="Build JPG Markov transition models.")
    parser.add_argument("--dataset", default=str(DATASET_REAL))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_seqs = []
    seg_bytes = defaultdict(list)
    sos_pool = []

    for p in tqdm(list(iter_jpegs(args.dataset)), desc="Parsing JPEGs"):
        data = read_bytes(p)
        mseq, segs, scans = parse_jpeg_segments(data)
        if mseq:
            marker_seqs.append(mseq)
        for mk, arrs in segs.items():
            seg_bytes[mk].extend(arrs)
        sos_pool.extend(scans)

    p_marker = build_bigram_markov(marker_seqs)
    p_segments = {mk: build_byte_markov(arrs) for mk, arrs in seg_bytes.items()}

    np.save(out_dir / "P_jpg_marker.npy", p_marker, allow_pickle=True)
    np.save(out_dir / "P_jpg_segments.npy", p_segments, allow_pickle=True)

    sos_dir = out_dir / "jpg_sos_pool"
    sos_dir.mkdir(parents=True, exist_ok=True)

    for i, scan in enumerate(sos_pool):
        (sos_dir / f"sos_{i:05d}.bin").write_bytes(scan)

    print("[+] Saved JPEG Markov models")


if __name__ == "__main__":
    main()
