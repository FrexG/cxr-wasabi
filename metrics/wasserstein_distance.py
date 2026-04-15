import numpy as np
import ot


def wasserstein_distance_pot(data1, data2, n):
    """
    Copied from WASABI-MRI Github repository
    https://github.com/BahramJafrasteh/wasabi-mri/blob/main/src/metrics.py
    """
    n_samples = n
    M = ot.dist(data1, data2)
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    return ot.emd2(a, b, M)
