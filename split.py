import numpy as np

from utils import NNumeric
from numpy.typing import NDArray
from typing import TypedDict, Any

class Split(TypedDict):
    X: NDArray[NNumeric]
    y: NDArray[NNumeric]

def class_non_iid_split(X: NDArray[NNumeric],
                        y: NDArray[NNumeric],
                        n_splits: int,
                        classes_per_split: int = 2) -> list[Split]:
    # Sort the classes
    ordered_indices: NDArray[np.int64] = np.argsort(y)

    # Divide the ordered indices into n_splits * classes_per_split parts
    cut_indices: list[NDArray[np.int64]] = np.array_split(
        ordered_indices, n_splits * classes_per_split)

    # Generate random pairs of these splits
    random_indices: NDArray[np.int64] = np.random.permutation(len(cut_indices))
    slipts_cuts: list[NDArray[np.int64]] = np.array_split(random_indices,
                                                          n_splits)

    # Combine the pairs of splits into one split
    splits: list[Split] = []
    for cuts in slipts_cuts:
        # Concatenate two split parts to form one combined split
        indices: NDArray[np.int64] = np.concatenate(
            [cut_indices[cuts[i]] for i in range(classes_per_split)])
        splits.append({'X': X[indices], 'y': y[indices]})

    return splits

def dirichlet_split(X: NDArray[NNumeric],
                    y: NDArray[NNumeric],
                    alpha: float = 0.5,
                    n_splits: int = 100) -> list[Split]:
    classes: NDArray[NNumeric] = np.unique(y)

    clients_samples: list[list[NDArray[np.int64]]] = [
        [] for _ in range(n_splits)
    ]

    # Partition the data for each class
    idx_by_class: dict[Any, NDArray[np.int64]] = {
        cls: np.where(y == cls)[0] for cls in classes
    }

    for cls in classes:
        # Get data points for this class
        idx: NDArray[np.int64] = idx_by_class[cls]

        # Draw proportions for each client using the Dirichlet distribution
        proportions: NDArray[np.float64] = np.random.dirichlet(
            np.repeat(alpha, n_splits))

        # Determine how many samples from this class each client will get
        class_splits: NDArray[np.int64] = np.floor(
            proportions * len(idx)).astype(int)

        # Ensure all samples are assigned by adjusting the split
        extra_samples: np.int64 = len(idx) - np.sum(class_splits)
        class_splits[:extra_samples] += 1

        # Shuffle the indices to randomize the selection
        np.random.shuffle(idx)

        # Assign the samples to each client
        current_idx: int = 0
        for i in range(n_splits):
            client_idx: NDArray[np.int64] = idx[
                np.arange(current_idx, current_idx + class_splits[i])
            ]
            clients_samples[i].append(client_idx)

            current_idx += class_splits[i]

    # Initialize client data partitions
    client_data: list[Split] = []

    for i in range(n_splits):
        client_indices: NDArray[np.int64] = np.concatenate(clients_samples[i])

        client_data.append({
            'X': X[client_indices],
            'y': y[client_indices]
        })

    return client_data
