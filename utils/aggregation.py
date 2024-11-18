import numpy as np

from scipy.spatial.distance import cdist
from typing import Union
from utils.utils import NumArray, bottom_indices, bottom_n, IntArray, FloatArray


def fed_avg(models_params: NumArray,
            weights: Union[NumArray, None] = None) -> NumArray:
    if len(models_params.shape) != 2:
        raise ValueError("models_params has to be a 2-D array")

    models_num: int = models_params.shape[0]
    if models_num == 0 or models_params.shape[1] == 0:
        return np.zeros((1, models_params.shape[1]))

    if weights is None:
        weights = np.repeat(1/models_num, models_num)
    weights = weights.astype("float")

    if models_num != weights.size:
        raise ValueError(
            "models_params.shape[0] and weights have to be of same length "
            "(%s != %s)" % (models_params.shape[0], weights.size)
        )

    return np.dot(weights, models_params)


def krum(models_params: NumArray,
         m: Union[float, int] = 0.3,
         weighting_mode: str = "uniform",
         data_sizes: Union[NumArray, None] = None) -> NumArray:
    if len(models_params.shape) != 2:
        raise ValueError("models_params has to be a 2-D array")

    models_num: int = models_params.shape[0]
    if models_num == 0 or models_params.shape[1] == 0:
        return np.zeros((1, models_params.shape[1]))
    elif models_num < 3:
        return fed_avg(models_params)

    # In case m is a proportion (default)
    if m < 1:
        m = models_num * m

    m = int(m)
    kept_models_num: int = models_num - m - 2
    if kept_models_num < 1:
        raise ValueError("models_num - m - 2 yielded < 1")

    distances: NumArray = cdist(models_params,
                                models_params,
                                metric='euclidean')

    scores: FloatArray = np.zeros(models_num, dtype="float")

    for i in range(models_num):
        # We use n_models + 1 because the distance with itself is always the
        # lowest (0)
        scores[i] = np.sum(bottom_n(distances[i], kept_models_num + 1))

    best_indices: IntArray = bottom_indices(scores, kept_models_num)

    weights: FloatArray
    if weighting_mode == "uniform":
        weights = np.repeat(1/best_indices.size,
                            best_indices.size).astype("float")
    elif weighting_mode == "data":
        if data_sizes is None:
            raise ValueError(
                "With weighting_mode='data', you have to provide data_sizes")
        elif data_sizes.size != models_num:
            raise ValueError("data_sizes.size has to be equals to "
                             "the number of provided models")

        weights = np.array([data_sizes[i] for i in best_indices], dtype="float")
        weights /= weights.sum()
    elif weighting_mode == "score":
        # The lowest score receives the highest weight
        weights = np.flip(np.arange(best_indices.size) + 1).astype("float")
        weights /= weights.sum()
    else:
        raise ValueError(f"Invalid weighting_mode '{weighting_mode}'")

    return fed_avg(models_params[best_indices], weights)
