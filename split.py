import numpy as np

from utils import NNumeric
from numpy.typing import NDArray
from typing import TypedDict, Union, Any


class Split(TypedDict):
    X: NDArray[NNumeric]
    y: NDArray[NNumeric]


def class_non_iid_split(X: NDArray[NNumeric],
                        y: NDArray[NNumeric],
                        n_splits: int,
                        n_classes_per_split: int = 2) -> list[Split]:
    classes: NDArray[NNumeric] = np.unique(y)
    n_classes: int = len(classes)
    n_splits_per_class: int = n_splits * n_classes_per_split // n_classes

    # For each class create a list with the required number splits introducing
    # some randomness in the inner-class indices selection
    classes_splits: list[list[NDArray[np.int64]]] = [
        np.array_split(
            np.random.permutation(np.where(y == cls)[0]),
            n_splits_per_class
        )
        for cls in classes
    ]
    # For each split we will keep track the classes that were already included
    splits_included_classes: list[list[int]] = [[] for _ in range(n_splits)]
    clients_samples: list[list[NDArray[np.int64]]] = [
        [] for _ in range(n_splits)
    ]

    i: int = 0
    n_remaining_splits: int = n_splits_per_class * n_classes
    for _ in range(n_splits_per_class * n_classes):
        not_included_classes: NDArray[NNumeric] = np.random.permutation(
            np.delete(classes, splits_included_classes[i])
        )

        for cls in not_included_classes:
            if classes_splits[cls]:
                splits_included_classes[i].append(cls)
                clients_samples[i].append(classes_splits[cls].pop())
                n_remaining_splits -= 1
                break

        i += 1
        if i == n_splits:
            i = 0

    # Remove the one level of depth in classes_splits with the not assigned
    # splits
    remaining_splits: list[NDArray[np.int64]] = [
        split for class_split in classes_splits for split in class_split
    ]

    i = 0
    # Include the remaining splits in those that ended up with less than the
    # required number of splits
    for split in remaining_splits:
        while len(clients_samples[i]) >= n_classes_per_split:
            i += 1

        # i should never go beyond n_splits because
        # n_splits_per_class * n_classes < n_splits
        clients_samples[i].append(split)

    # Initialize client data partitions
    client_data: list[Split] = []

    for i in range(n_splits):
        client_indices: NDArray[np.int64] = np.concatenate(clients_samples[i])

        client_data.append({
            'X': X[client_indices],
            'y': y[client_indices]
        })

    return client_data


def dirichlet_split(X: NDArray[NNumeric],
                    y: NDArray[NNumeric],
                    n_splits: int,
                    alpha: float = 0.5,
                    split_min_size: Union[int, None] = None) -> list[Split]:
    classes: NDArray[NNumeric] = np.unique(y)
    is_index_available: NDArray[np.bool_] = np.repeat(True, len(y))
    clients_samples: list[list[NDArray[np.int64]]] = []

    if split_min_size is None:
        clients_samples = [[] for _ in range(n_splits)]
    else:
        initial_indices: NDArray[np.int64] = np.random.choice(
            np.arange(len(y)),
            size=split_min_size * n_splits,
            replace=False
        )

        clients_samples = [
            [split] for split in np.array_split(initial_indices, n_splits)
        ]

        is_index_available[initial_indices] = False

    # Partition the data for each class
    idx_by_class: dict[Any, NDArray[np.int64]] = {
        cls: np.where(np.logical_and(y == cls, is_index_available))[0]
        for cls in classes
    }

    for cls in classes:
        # Get data points for this class
        idx: NDArray[np.int64] = idx_by_class[cls]

        # Draw proportions for each client using the Dirichlet distribution
        proportions: NDArray[np.float64] = np.random.dirichlet(
            np.repeat(alpha, n_splits)
        )

        # Determine how many samples from this class each client will get
        class_splits: NDArray[np.int64] = np.floor(
            proportions * len(idx)
        ).astype(int)

        # Ensure all samples are assigned by adjusting the split
        extra_samples: np.int64 = len(idx) - np.sum(class_splits)
        class_splits[np.random.choice(len(class_splits),
                                      extra_samples,
                                      replace=False)] += 1

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
