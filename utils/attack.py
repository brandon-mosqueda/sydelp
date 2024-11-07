import numpy as np

from typing import Union
from numpy.typing import NDArray
from utils.utils import ModelParams, NNumeric


def random_gaussian_model(reference: ModelParams,
                          mean: float = 0,
                          sd: float = 1) -> ModelParams:
    return [
        np.random.normal(loc=mean, scale=sd, size=layer.shape)
        for layer in reference
    ]


def flip_labels(y: NDArray[NNumeric],
                target_label_1: Union[np.int_, int],
                target_label_2: Union[np.int_, int]) -> None:
    original_positions: NDArray[np.bool_] = y == target_label_1
    target_positions: NDArray[np.bool_] = y == target_label_2

    y[original_positions] = target_label_2
    y[target_positions] = target_label_1
