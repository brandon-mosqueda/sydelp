import numpy as np

from numpy.typing import NDArray
from typing import Union
from utils.utils import Weights, bottom_indices, bottom_n


def fed_avg(weights: list[Weights]) -> Weights:
    if not weights:
        return []

    avg_weights: Weights = []

    for i in range(len(weights[0])):
        avg_weights.append(np.mean([w[i] for w in weights], axis=0))

    return avg_weights


def weights_distance(x: Weights, y: Weights) -> np.float128:
    dist: np.float128 = np.sum(
        [np.sum((x_i - y_i)**2, dtype="float128") for x_i, y_i in zip(x, y)]
    )

    return np.sqrt(dist)


def weights_distance_matrix(weights: list[Weights]) -> NDArray[np.float128]:
    n: int = len(weights)
    matrix: NDArray[np.float128] = np.zeros((n, n), dtype="float128")

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = weights_distance(weights[i], weights[j])
            matrix[j, i] = matrix[i, j]

    return matrix


def krum(weights: list[Weights], m: Union[float, int] = 0.3) -> Weights:
    if len(weights) < 3:
        return fed_avg(weights)

    if m < 1:
        m = len(weights) * m

    n: int = len(weights)
    m = min(int(m), n - 2)
    n_models: int = n - m - 2

    distances: NDArray[np.float128] = weights_distance_matrix(weights)
    scores: NDArray[np.float128] = np.zeros(n, dtype="float128")

    for i in range(n):
        # We use n_models + 1 because the distance with itself is always the
        # lowest (0)
        scores[i] = np.sum(bottom_n(distances[i], n_models + 1))

    best_indices: NDArray[np.int64] = bottom_indices(scores, n - m)
    aggregated_model: Weights = fed_avg([weights[i] for i in best_indices])

    return aggregated_model
