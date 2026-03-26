import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"

OUT_DIR = BASE_DIR / "matrici_alias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_SEGMENTS_PATH = BASE_DIR / "P_jpg_segments.npy"

# ============================================================
# ALIAS METHOD HELPERS
# ============================================================
def build_alias_table(p):
    """
    Costruisce alias table per una distribuzione discreta p (len=256)
    """
    n = len(p)
    prob  = np.zeros(n, dtype=np.float32)
    alias = np.zeros(n, dtype=np.uint8)

    scaled = p * n
    small = []
    large = []

    for i in range(n):
        if scaled[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        s = small.pop()
        l = large.pop()

        prob[s] = scaled[s]
        alias[s] = l

        scaled[l] -= (1.0 - scaled[s])
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for i in small + large:
        prob[i] = 1.0
        alias[i] = i

    return prob, alias


def build_alias_markov(P):
    """
    Costruisce alias tables per una matrice di transizione 256x256
    """
    prob  = np.zeros((256, 256), dtype=np.float32)
    alias = np.zeros((256, 256), dtype=np.uint8)

    for y in range(256):
        prob[y], alias[y] = build_alias_table(P[y])

    return prob, alias


def dict_to_matrix(P_dict):
    """
    Converte {a: [(b,p), ...]} in matrice 256x256 normalizzata
    """
    M = np.zeros((256, 256), dtype=np.float32)

    for a, transitions in P_dict.items():
        for b, p in transitions:
            M[a, b] = p

    row_sum = M.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    M /= row_sum

    return M

# ============================================================
# MAIN
# ============================================================
def main():
    print("[+] Loading JPG segment Markov models")
    P_segments_raw = np.load(P_SEGMENTS_PATH, allow_pickle=True).item()

    # ---- marker indexing (NO DICT A RUNTIME) ----
    markers = sorted(P_segments_raw.keys())
    marker_to_idx = {m: i for i, m in enumerate(markers)}
    idx_to_marker = np.array(markers, dtype=np.uint8)

    num_markers = len(markers)
    print(f"[+] Found {num_markers} JPG markers")

    # ---- allocate PURE numpy arrays ----
    alias_prob = np.zeros((num_markers, 256, 256), dtype=np.float32)
    alias_idx  = np.zeros((num_markers, 256, 256), dtype=np.uint8)

    # ---- build alias tables ----
    for marker, P_dict in P_segments_raw.items():
        mi = marker_to_idx[marker]
        print(f"    marker 0x{marker:02X}")

        P = dict_to_matrix(P_dict)
        prob, idx = build_alias_markov(P)

        alias_prob[mi] = prob
        alias_idx[mi]  = idx

    # ---- save EVERYTHING as numeric arrays ----
    np.save(OUT_DIR / "P_jpg_segments_alias_prob.npy", alias_prob)
    np.save(OUT_DIR / "P_jpg_segments_alias_idx.npy",  alias_idx)
    np.save(OUT_DIR / "P_jpg_segments_marker_map.npy", idx_to_marker)

    print("[+] JPG alias tables saved correctly")

# ============================================================
if __name__ == "__main__":
    main()
