from numba import njit
import numpy as np


@njit
def _sample_alias(prob_row, alias_row):
    i = np.random.randint(256)
    r = np.random.random()
    if r < prob_row[i]:
        return i
    return alias_row[i]


@njit
def _generate_bytes_markov_alias(
    alias_prob,
    alias_idx,
    length,
    start_byte=-1
):
    if start_byte < 0:
        y = np.random.randint(256)
    else:
        y = start_byte

    out = np.empty(length, dtype=np.uint8)
    out[0] = y

    for i in range(1, length):
        y = _sample_alias(alias_prob[y], alias_idx[y])
        out[i] = y

    return out


@njit
def _generate_bytes_markov_matrix(P, length, start_byte=-1):
    if start_byte < 0:
        y = np.random.randint(256)
    else:
        y = start_byte

    out = np.empty(length, dtype=np.uint8)
    out[0] = y

    for i in range(1, length):
        r = np.random.random()
        acc = 0.0
        for x in range(256):
            acc += P[y, x]
            if r <= acc:
                y = x
                break
        out[i] = y

    return out


def generate_bytes_markov(
    model: np.ndarray | None = None,
    alias_idx: np.ndarray | int | None = None,
    length: int | None = None,
    *,
    P: np.ndarray | None = None,
    start_byte: int | None = None
) -> bytes:
    if P is not None:
        model = P
    if length is None and isinstance(alias_idx, (int, np.integer)):
        length = int(alias_idx)
        alias_idx = None
    if model is None or length is None:
        raise TypeError("generate_bytes_markov requires a model and a length")
    if length <= 0:
        return b""

    sb = -1 if start_byte is None else start_byte
    if alias_idx is None:
        arr = _generate_bytes_markov_matrix(model, length, sb)
    else:
        arr = _generate_bytes_markov_alias(model, alias_idx, length, sb)
    return arr.tobytes()
