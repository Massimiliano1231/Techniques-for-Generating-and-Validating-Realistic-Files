from numba import njit
import numpy as np

@njit
def sample_alias(prob_row, alias_row):
    i = np.random.randint(256)
    r = np.random.random()
    if r < prob_row[i]:
        return i
    else:
        return alias_row[i]

@njit
def generate_bytes_markov_alias_numba(
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
        y = sample_alias(alias_prob[y], alias_idx[y])
        out[i] = y

    return out

def generate_bytes_markov(
    alias_prob,
    alias_idx,
    length,
    start_byte=None
) -> bytes:
    sb = -1 if start_byte is None else start_byte
    arr = generate_bytes_markov_alias_numba(
        alias_prob,
        alias_idx,
        length,
        sb
    )
    return arr.tobytes()





"""
import numpy as np
from numba import njit

@njit
def generate_bytes_markov_numba(P, length, start_byte=-1):
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


def generate_bytes_markov(P, length, start_byte=None) -> bytes:
    sb = -1 if start_byte is None else start_byte
    arr = generate_bytes_markov_numba(P, length, sb)
    return arr.tobytes()





def generate_bytes_markov(
    P: np.ndarray,
    length: int,
    start_byte: int | None = None
) -> bytes:
 

    if start_byte is None:
        y = np.random.randint(0, 256)
    else:
        y = start_byte

    out = bytearray()
    out.append(y)

    for _ in range(length - 1):
        probs = P[y]
        x = np.random.choice(256, p=probs)
        out.append(x)
        y = x

    return bytes(out)
"""
