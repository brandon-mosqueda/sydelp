import numpy as np

from scipy.spatial.distance import cdist
from typing import Union
from utils.utils import bottom_indices, bottom_n
from utils.typing import IntArray, FloatArray, BoolArray


def fed_avg(models_params: FloatArray,
            weights: Union[FloatArray, None] = None) -> FloatArray:
    if len(models_params.shape) != 2:
        raise ValueError("models_params has to be a 2-D array")

    models_num: int = models_params.shape[0]
    if models_num == 0 or models_params.shape[1] == 0:
        return np.zeros((1, models_params.shape[1]), dtype="float")

    if weights is None:
        weights = np.repeat(1/models_num, models_num)
    weights = weights.astype("float")

    if models_num != weights.size:
        raise ValueError(
            "models_params.shape[0] and weights have to be of same length "
            "(%s != %s)" % (models_params.shape[0], weights.size)
        )

    return np.dot(weights, models_params)


def krum_selection(models_params: FloatArray,
                   m: Union[float, int] = 0.3) -> BoolArray:
    if len(models_params.shape) != 2:
        raise ValueError("models_params has to be a 2-D array")

    models_num: int = models_params.shape[0]
    if models_num == 0 or models_params.shape[1] == 0:
        return np.array([], dtype="int")
    elif models_num < 3:
        return np.arange(models_num, dtype="int")

    # In case m is a proportion (default)
    if m < 1:
        m = models_num * m

    m = int(m)
    closest_models_num: int = models_num - m - 2
    if closest_models_num < 1:
        raise ValueError("models_num - m - 2 yielded < 1")

    distances: FloatArray = cdist(models_params,
                                  models_params,
                                  metric='euclidean')

    scores: FloatArray = np.zeros(models_num, dtype="float")

    for i in range(models_num):
        # We use closest_models_num + 1 because the distance with itself is
        # always included for since it is the lowest (0)
        scores[i] = np.sum(bottom_n(distances[i], closest_models_num + 1))

    selected_idx: BoolArray = np.repeat(False, models_num)
    selected_idx[bottom_indices(scores, closest_models_num)] = True

    return selected_idx


def krum(models_params: FloatArray,
         m: Union[float, int] = 0.3,
         weighting_mode: str = "uniform",
         data_sizes: Union[IntArray, None] = None) -> FloatArray:
    selected_idx: BoolArray = krum_selection(models_params, m)
    models_num: int = models_params.shape[0]

    weights: FloatArray
    if weighting_mode == "uniform":
        weights = np.repeat(1/selected_idx.size,
                            selected_idx.size).astype("float")
    elif weighting_mode == "data":
        if data_sizes is None:
            raise ValueError(
                "With weighting_mode='data', you have to provide data_sizes")
        elif data_sizes.size != models_num:
            raise ValueError("data_sizes.size has to be equals to "
                             "the number of provided models")

        weights = np.array(data_sizes[selected_idx], dtype="float")
        weights /= weights.sum()
    else:
        raise ValueError(f"Invalid weighting_mode '{weighting_mode}'")

    return fed_avg(models_params[selected_idx], weights)
