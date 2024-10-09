import numpy as np

from utils import NNumeric
from numpy.typing import NDArray
from typing import TypedDict

class Split(TypedDict):
    X: NDArray[NNumeric]
    y: NDArray[NNumeric]

def non_idd_split(X: NDArray[NNumeric],
                  y: NDArray[NNumeric],
                  splits_num: int) -> list[Split]:
    # Sort the classes
    ordered_indices: NDArray[np.int64] = np.argsort(y)

    # Divide the ordered indices into splits_num * 2 parts
    cut_indices: list[NDArray[np.int64]] = np.array_split(ordered_indices,
                                                          splits_num * 2)

    # Generate random pairs of these splits
    random_indices: NDArray[np.int64] = np.random.permutation(len(cut_indices))
    splits_pairs: list[NDArray[np.int64]] = np.array_split(random_indices,
                                                           splits_num)

    # Combine the pairs of splits into one split
    splits: list[Split] = []
    for pair in splits_pairs:
        # Concatenate two split parts to form one combined split
        indices: NDArray[np.int64] = np.concatenate([cut_indices[pair[0]],
                                                     cut_indices[pair[1]]])
        splits.append({'X': X[indices], 'y': y[indices]})

    return splits
