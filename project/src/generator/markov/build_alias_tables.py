import numpy as np
from pathlib import Path

# ==========================================================
# PATH
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"

ALIAS_DIR = BASE_DIR / "matrici_alias"
ALIAS_DIR.mkdir(parents=True, exist_ok=True)

MARKOV_PATHS = {
    "txt":  BASE_DIR / "P_txt.npy",
    "pdf":  BASE_DIR / "P_pdf.npy",
    "docx": BASE_DIR / "P_docx.npy",
}

# ==========================================================
# ALIAS METHOD (OFFLINE)
# ==========================================================
def build_alias_table(p):
    """
    p: array di probabilità (somma = 1), shape (256,)
    ritorna:
      prob  -> float32, shape (256,)
      alias -> int32,   shape (256,)
    """
    n = len(p)
    prob  = np.zeros(n, dtype=np.float32)
    alias = np.zeros(n, dtype=np.int32)

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

    for i in large + small:
        prob[i] = 1.0
        alias[i] = i

    return prob, alias


def build_alias_markov(P):
    """
    P: matrice Markov (256x256)
    ritorna:
      prob  -> (256,256)
      alias -> (256,256)
    """
    prob  = np.zeros((256, 256), dtype=np.float32)
    alias = np.zeros((256, 256), dtype=np.int32)

    for y in range(256):
        prob[y], alias[y] = build_alias_table(P[y])

    return prob, alias


# ==========================================================
# MAIN
# ==========================================================
def main():
    for fmt, path in MARKOV_PATHS.items():
        print(f"[+] Building alias tables for {fmt.upper()}")

        P = np.load(path)
        assert P.shape == (256, 256)
        assert np.allclose(P.sum(axis=1), 1.0)

        prob, alias = build_alias_markov(P)

        out_prob  = ALIAS_DIR / f"P_{fmt}_alias_prob.npy"
        out_alias = ALIAS_DIR / f"P_{fmt}_alias_idx.npy"

        np.save(out_prob, prob)
        np.save(out_alias, alias)

        print(f"    saved -> {out_prob.name}, {out_alias.name}")

    print("[+] Done. Alias tables generated.")


if __name__ == "__main__":
    main()
