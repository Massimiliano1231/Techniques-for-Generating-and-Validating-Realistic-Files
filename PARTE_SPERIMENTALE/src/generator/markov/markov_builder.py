import numpy as np

def normalize_rows(C: np.ndarray) -> np.ndarray:
    P = np.zeros_like(C, dtype=np.float64)

    for y in range(256):
        row_sum = C[y].sum()
        if row_sum > 0:
            P[y] = C[y] / row_sum
        else:
            P[y] = 1.0 / 256.0

    return P
