import numpy as np

from numpy.typing import NDArray
from typing import Union
from utils.utils import ModelParams, bottom_indices, bottom_n


def fed_avg(models_params: list[ModelParams]) -> ModelParams:
    if not models_params:
        return []

    avg_params: ModelParams = []

    for i in range(len(models_params[0])):
        avg_params.append(np.mean([w[i] for w in models_params], axis=0))

    return avg_params


def model_params_distance(x: ModelParams, y: ModelParams) -> np.float128:
    dist: np.float128 = np.sum(
        [np.sum((x_i - y_i)**2, dtype="float128") for x_i, y_i in zip(x, y)]
    )

    return np.sqrt(dist)


def model_params_distance_matrix(
        models_params: list[ModelParams]) -> NDArray[np.float128]:
    n: int = len(models_params)
    matrix: NDArray[np.float128] = np.zeros((n, n), dtype="float128")

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = model_params_distance(models_params[i],
                                                 models_params[j])
            matrix[j, i] = matrix[i, j]

    return matrix


def krum(models_params: list[ModelParams],
         m: Union[float, int] = 0.3) -> ModelParams:
    if len(models_params) < 3:
        return fed_avg(models_params)

    if m < 1:
        m = len(models_params) * m

    n: int = len(models_params)
    m = min(int(m), n - 2)
    n_models: int = n - m - 2

    distances: NDArray[np.float128] = model_params_distance_matrix(models_params)
    scores: NDArray[np.float128] = np.zeros(n, dtype="float128")

    for i in range(n):
        # We use n_models + 1 because the distance with itself is always the
        # lowest (0)
        scores[i] = np.sum(bottom_n(distances[i], n_models + 1))

    best_indices: NDArray[np.int64] = bottom_indices(scores, n - m)
    aggregated_model: ModelParams = fed_avg([models_params[i] for i in best_indices])

    return aggregated_model
