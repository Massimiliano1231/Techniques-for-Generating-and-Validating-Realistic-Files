import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))


from formats.jpg_helper import (
    iter_jpegs, read_bytes, parse_jpeg_segments,
    build_bigram_markov, build_byte_markov
)

DATASET_REAL = PROJECT_ROOT / "data" / "detector" / "datasets" / "jpg data" / "JPG-total"
OUT_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

marker_seqs = []
seg_bytes = defaultdict(list)
sos_pool = []

for p in tqdm(list(iter_jpegs(DATASET_REAL)), desc="Parsing JPEGs"):
    data = read_bytes(p)
    mseq, segs, scans = parse_jpeg_segments(data)
    if mseq:
        marker_seqs.append(mseq)
    for mk, arrs in segs.items():
        seg_bytes[mk].extend(arrs)
    sos_pool.extend(scans)

P_marker = build_bigram_markov(marker_seqs)
P_segments = {mk: build_byte_markov(arrs) for mk, arrs in seg_bytes.items()}

np.save(OUT_DIR / "P_jpg_marker.npy", P_marker, allow_pickle=True)
np.save(OUT_DIR / "P_jpg_segments.npy", P_segments, allow_pickle=True)
SOS_DIR = OUT_DIR / "jpg_sos_pool"
SOS_DIR.mkdir(parents=True, exist_ok=True)

for i, scan in enumerate(sos_pool):
    (SOS_DIR / f"sos_{i:05d}.bin").write_bytes(scan)


print("[+] Saved JPEG Markov models")
