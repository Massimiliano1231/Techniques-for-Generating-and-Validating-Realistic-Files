import numpy as np


def make_objective_default(scores_real, scores_rand, alpha, beta):
    """
    Per metriche 'distanza' (JSD, TVD, L1):
      -> REAL dovrebbero avere score <= soglia
      -> RANDOM score > soglia
    """
    def fn_fp_rates(threshold):
        fn = np.sum(scores_real > threshold) / len(scores_real)
        fp = np.sum(scores_rand <= threshold) / len(scores_rand)
        return fn, fp

    def objective_soft(x):
        t = x[0]
        fn, fp = fn_fp_rates(t)
        return alpha * fn + beta * fp

    return objective_soft, fn_fp_rates



def make_objective_cosine(scores_real, scores_rand, alpha, beta):
    """
    Cosine similarity:
      -> REAL dovrebbero avere score >= soglia
      -> RANDOM score < soglia
    """
    def fn_fp_rates(threshold):
        fn = np.sum(scores_real < threshold) / len(scores_real)
        fp = np.sum(scores_rand >= threshold) / len(scores_rand)
        return fn, fp

    def objective_soft(x):
        t = x[0]
        fn, fp = fn_fp_rates(t)
        return alpha * fn + beta * fp

    return objective_soft, fn_fp_rates




def make_objective_entropy_band(scores_real, scores_rand, alpha, beta):
    """
    Entropy:
      -> REAL dovrebbero stare dentro [low, high]
      -> RANDOM fuori (idealmente).
    """
    def fn_fp_rates_band(x):
        low, high = x
        if low > high:
            low, high = high, low
        fn = np.sum((scores_real < low) | (scores_real > high)) / len(scores_real)
        fp = np.sum((scores_rand >= low) & (scores_rand <= high)) / len(scores_rand)
        return fn, fp, low, high

    def objective_soft(x):
        fn, fp, _, _ = fn_fp_rates_band(x)
        return alpha * fn + beta * fp

    return objective_soft, fn_fp_rates_band
