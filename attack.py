import numpy as np

from utils import Weights


def random_gaussian_weights(reference: Weights,
                            mean: float = 0,
                            sd: float = 1) -> Weights:
    return [
        np.random.normal(loc=mean, scale=sd, size=layer.shape)
        for layer in reference
    ]
