import numpy as np

from utils.utils import NumArray


def random_gaussian_model(size: int,
                          mean: float = 0,
                          sd: float = 1) -> NumArray:
    return np.random.normal(loc=mean, scale=sd, size=size)
