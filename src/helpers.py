import numpy as np


def sample_Xi():
    mu, sigma = 0, 1
    return np.random.normal(mu, sigma, 1)