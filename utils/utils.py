import progressbar

import keras
import numpy as np

from keras import models
from typing import Union, Any
from json import load as load_json
from numpy.typing import NDArray
from utils.typing import NumArray, IntArray, KerasModel


def print_array(x: NDArray, n: int = 5, digits: int = 4) -> None:
    if np.issubdtype(x.dtype, np.number):
        x = x.round(digits)

    print_dots: bool = len(x) > n
    x = x[:min(n, len(x))]

    print('[', end='')
    print(", ".join(map(str, x)), end='')
    if print_dots:
        print(', ...', end='')
    print(']')


def summary(obj: object,
            max_depth: int = 100,
            depth: int = 0,
            digits: int = 4,
            max_items: int = 100) -> None:
    indent_str: str = '...' * depth

    if depth > max_depth:
        print(f"...")
        return
    elif isinstance(obj, list):
        print(f"List of {len(obj)}:")

        if depth == max_depth:
            return

        print_num: int = min(len(obj), max_items)
        for i in range(print_num):
            print(f"{indent_str}$[{i}]", end=' ')
            summary(obj[i], max_depth, depth + 1)

        if print_num < len(obj):
            print(f"{indent_str}$[...]")
    elif isinstance(obj, np.ndarray):
        if len(obj.shape) == 1:
            print(f"Numpy of {len(obj)}:", end=' ')
            print_array(obj, n=5, digits=digits)
        else:
            print(f"Numpy of {obj.shape}:")
            if depth == max_depth:
                return

            print_num: int = min(len(obj), max_items)
            for i in range(print_num):
                print(f"{indent_str}$[{i}]", end=' ')
                summary(obj[i], max_depth, depth + 1)

            if print_num < len(obj):
                print(f"{indent_str}$[...]")
    elif isinstance(obj, dict):
        print(f"Dict of {len(obj)}:")
        if depth == max_depth:
            return

        for key, value in obj.items():
            print(f"{indent_str}${key}: ", end='')
            summary(value, max_depth, depth + 1)
    else:
        if isinstance(obj, (int, float)):
            obj = round(obj, digits)

        # Base case: primitive types or other objects
        print(f"{type(obj).__name__}: {repr(obj)}")


class MyProgressBar(progressbar.ProgressBar):
    current: int = 0

    def next(self) -> None:
        self.current = self.current + 1

        super().update(self.current)


def progress_bar(rounds: int) -> MyProgressBar:
    widgets: list = [
        progressbar.Bar(marker='#', left='[', right=']'),
        ' ', progressbar.Counter(), f'/{rounds}',
        ' ', progressbar.Timer()
    ]

    return MyProgressBar(widgets=widgets, max_value=rounds)


# Count the number of apperances
def count(x: Union[NDArray, list, tuple]) -> dict:
    np_counts: tuple[NDArray, NDArray] = np.unique(x, return_counts=True)

    return {cls: num for cls, num in zip(np_counts[0], np_counts[1])}


def top_indices(x: NumArray, n: int) -> IntArray:
    if n < 0 or n > x.shape[0]:
        raise ValueError("n should be 0 <= n <= x.shape")

    if n == 0:
        return np.array([], dtype="int")

    return np.flip(np.argsort(x))[:n]


def top_n(x: NumArray, n: int = 1) -> NumArray:
    return x[top_indices(x, n)]


def bottom_indices(x: NumArray, n: int = 1) -> IntArray:
    if n < 0 or n > x.shape[0]:
        raise ValueError("n should be 0 <= n <= x.shape")

    if n == 0:
        return np.array([], dtype="int")

    return np.argsort(x)[:n]


def bottom_n(x: NumArray, n: int = 1) -> NumArray:
    return x[bottom_indices(x, n)]


def elapsed_time(start: float, end: float, units: str = "mins") -> float:
    elapsed: float = end - start

    if units == "mins":
        elapsed /= 60
    elif units == "hours":
        elapsed /= 3600

    return elapsed


def remove_indices(x: list, indices: list[int]) -> list:
    indices_set: set = set(indices)

    return [
        item for i, item in enumerate(x)
        if i not in indices_set
    ]


def replicate_model(model: KerasModel,
                    n: int = 5,
                    same_params: bool = True,
                    compiled: bool = True) -> list[KerasModel]:
    model_list: list[KerasModel] = [models.clone_model(model) for _ in range(n)]

    if same_params:
        [mod.set_weights(model.get_weights()) for mod in model_list]

    if compiled:
        for mod in model_list:
            mod.compile(
                optimizer=keras.optimizers.deserialize(
                    keras.optimizers.serialize(model.optimizer)),
                loss=model.loss
            )

    return model_list


def read_json(file: str) -> dict:
    with open(file) as json_file:
        return load_json(json_file)


def as_name(x: Any):
    return str(x).lower().replace(' ', '_')


def set_list_weights_in_array(weights: list[NumArray], arr: NumArray) -> None:
    idx = 0
    for layer in weights:
        flat_size = layer.size
        # Fill the array in place
        arr[idx:idx + flat_size] = layer.flatten()
        idx += flat_size


def get_flatten_weights(model: KerasModel) -> NumArray:
    weights = model.get_weights()
    total_size = sum(w.size for w in weights)

    # Pre-allocate a single array with the necessary size
    flat_weights = np.empty(total_size, dtype="float32")

    # Fill the flat_weights array in place
    idx = 0
    for w in weights:
        flat_size = w.size
        flat_weights[idx:idx + flat_size] = w.flatten()
        idx += flat_size

    return flat_weights


def flatten_to_original(weights: NumArray,
                        ref_model: KerasModel) -> list[NumArray]:
    ref_weights: list[NumArray] = ref_model.get_weights()
    new_weights: list[NumArray] = []
    index = 0

    # Directly reshape each section of `weights` without additional slicing
    for layer in ref_weights:
        num_elements = layer.size
        new_weights.append(weights[index:index + num_elements]
                           .reshape(layer.shape))
        index += num_elements

    return new_weights


def set_flatten_weights(model: KerasModel, weights: NumArray) -> None:
    model.set_weights(flatten_to_original(weights, model))


def replace_by(x: Union[NDArray, list],
               values_mapping: dict,
               default_value=None) -> Union[NDArray, list]:
    res: Union[NDArray, list] = [values_mapping.get(key, default_value)
                                 for key in x]
    if isinstance(x, np.ndarray):
        res = np.array(res)

    return res
