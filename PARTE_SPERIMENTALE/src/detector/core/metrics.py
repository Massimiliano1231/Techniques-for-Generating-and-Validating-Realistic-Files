import numpy as np
from scipy.spatial.distance import cosine

def entropy(p):
    m = p>0
    return float(-np.sum(p[m]*np.log2(p[m])))

def _kl(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        m = (a>0)
        return float(np.sum(a[m] * (np.log2(a[m]) - np.log2(b[m]))))

def jsd(p,q):
    m = 0.5*(p+q)
    return 0.5*(_kl(p,m)+_kl(q,m))

def tvd(p,q): return float(0.5*np.sum(np.abs(p-q)))
def l1_distance(p,q): return float(np.sum(np.abs(p-q)))



def cosine_sim(p,q):
    if p.sum()==0 or q.sum()==0: return 0.0
    val = 1.0 - cosine(p,q)
    return 0.0 if np.isnan(val) else float(val)

def compute_metrics(p, mean_bfd):
    return {
        "JSD": jsd(p, mean_bfd),
        "TVD": tvd(p, mean_bfd),
        "L1":  l1_distance(p, mean_bfd),
        "Cosine":   cosine_sim(p, mean_bfd),
        "Entropy":  entropy(p),
    }
