import progressbar

import numpy as np
import keras.optimizers as optimizers  # type: ignore

from json import load as load_json
from keras import models
from keras.src.models import Model as KerasModel
from numpy.typing import NDArray
from typing import Union, Any

NNumeric = Union[
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64, np.float128
]

Int = Union[np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
            np.int8, np.int16, np.int32, np.int64, int]

Float = Union[float, np.float16, np.float32, np.float64, np.float128]

Weights = list[NDArray[NNumeric]]


def print_array(x: NDArray[np.generic], n: int = 5, digits: int = 4) -> None:
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def next(self) -> None:
        self.current = self.current + 1

        super().update(self.current)


def progress_bar(rounds: int) -> MyProgressBar:
    widgets: list = [
        progressbar.Bar(marker='#', left='[', right=']'),
        ' ', progressbar.Counter(), f'/{rounds}',
        ' ', progressbar.Timer()
    ]

    bar: MyProgressBar = MyProgressBar(widgets=widgets, max_value=rounds)

    return bar


# Count the number of apperances
def count(x: Union[NDArray, list, tuple]) -> dict:
    np_counts: tuple[NDArray, NDArray] = np.unique(x, return_counts=True)

    return {cls: num for cls, num in zip(np_counts[0], np_counts[1])}


def top_indices(x: NDArray[NNumeric], n: int) -> NDArray[np.int64]:
    if n < 0 or n > x.shape[0]:
        raise ValueError("n should be 0 <= n <= x.shape")

    if n == 0 or x.shape[0] == 0:
        return np.array([], dtype="int64")

    n = min(n, x.shape[0])

    return np.flip(np.argsort(x))[:n]


def top_n(x: NDArray[NNumeric], n: int = 1) -> NDArray[NNumeric]:
    return x[top_indices(x, n)]


def bottom_indices(x: NDArray[NNumeric], n: int = 1) -> NDArray[np.int64]:
    if n < 0 or n > x.shape[0]:
        raise ValueError("n should be 0 <= n <= x.shape")

    if n == 0 or x.shape[0] == 0:
        return np.array([], dtype="int64")

    n = min(n, x.shape[0])

    return np.argsort(x)[:n]


def bottom_n(x: NDArray[NNumeric], n: int = 1) -> NDArray[NNumeric]:
    return x[bottom_indices(x, n)]


def elapsed_time(start: float, end: float, units: str = "mins") -> float:
    elapsed: float = end - start

    if units == "mins":
        elapsed /= 60
    elif units == "hours":
        elapsed /= 3600

    return elapsed


def remove_indices(x: list, indices: list) -> list:
    indices_set: set = set(indices)

    return [
        item for i, item in enumerate(x)
        if i not in indices_set
    ]


def replicate_model(model: KerasModel,
                    n: int = 5,
                    same_weights: bool = True,
                    compiled: bool = True) -> list[KerasModel]:
    model_list: list[KerasModel] = [models.clone_model(model) for _ in range(n)]

    if same_weights:
        [mod.set_weights(model.get_weights()) for mod in model_list]

    if compiled:
        for mod in model_list:
            mod.compile(optimizer=optimizers.deserialize(
                            optimizers.serialize(model.optimizer)),
                        loss=model.loss)

    return model_list


def read_json(file: str) -> dict:
    with open(file) as json_file:
        return load_json(json_file)


def as_name(x: Any):
    return str(x).lower().replace(' ', '_')
