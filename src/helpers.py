import numpy as np


def sample_Xi(mu=0, sigma=1):
    """
    samples the normal distribution
    :return: float [0, 1]
    """
    return np.random.normal(mu, sigma, 1)
