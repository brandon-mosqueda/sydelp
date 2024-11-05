import utils.utils as utils
import random

import numpy as np

from utils.split import *
from unittest import TestCase
from typing import Union
from numpy.typing import NDArray
from utils.split import Split
from utils.utils import NNumeric
from sklearn.datasets import load_iris, load_digits
from numpy.testing import assert_array_equal

class TestSplit(TestCase):
    iris: Split
    digits: Split
    seed: int = 5445

    def setUp(self):
        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        X, y = load_iris(return_X_y=True)
        self.iris = {"X": np.array(X), "y": np.array(y)}

        x_test, y_test = load_digits(return_X_y=True)
        self.digits = {"X": np.array(x_test), "y": np.array(y_test)}

    def assert_splits(self,
                      splits: list[Split],
                      original_y: NDArray[NNumeric],
                      expected_n_splits: int):
        self.assertEqual(len(splits),
                         expected_n_splits,
                         f"Should create {expected_n_splits} splits")

        for split in splits:
            self.assertIsInstance(split, dict, "Each split should be a dict")
            self.assertIn('X', split, "Each split should have 'X' key")
            self.assertIn('y', split, "Each split should have 'y' key")
            self.assertEqual(
                split['X'].shape[0],
                split['y'].shape[0],
                "'X' and 'y' has to have the same length in each split"
            )

        total_samples: int = sum([split['y'].shape[0] for split in splits])
        original_length: int = original_y.shape[0]
        self.assertEqual(original_length,
                         total_samples,
                         "Total split's samples should match original data")

        original_counts: dict = utils.count(original_y)
        splits_counts: dict = {cls: 0 for cls in original_counts}

        for split in splits:
            counts: dict = utils.count(split['y'])
            for key in counts:
                splits_counts[key] += counts[key]

        self.assertDictEqual(
            original_counts,
            splits_counts,
            "The original proportion of classes should be preserved"
        )

    def assert_class_non_iid_splits(self,
                                    splits: list[Split],
                                    n_classes_per_split: int):
        classes_per_split = np.array(
            [len(np.unique(split['y'])) for split in splits]
        )

        result: NDArray[np.bool_] = np.logical_or(
            classes_per_split == n_classes_per_split,
            classes_per_split == n_classes_per_split - 1
        )

        self.assertTrue(
            np.all(result),
            "Each non-iid split should have n_classes_per_split "
            + "or n_classes_per_split - 1 different classes"
        )

    def assert_dirichlet_splits(self,
                                splits: list[Split],
                                split_min_size: Union[int, None]):
        if split_min_size is not None:
            splits_lengths: NDArray[np.int64] = np.array(
                [split['y'].shape[0] for split in splits]
            )

            self.assertTrue(
                np.all(splits_lengths >= split_min_size),
                "All splits should have at least split_min_size size"
            )

    def assert_same_splits(self, x: list[Split], y: list[Split]):
        self.assertEqual(len(x), len(y), "Splits have to have the same length")

        for s, ss in zip(x, y):
            assert_array_equal(s['X'], ss['X'])
            assert_array_equal(s['y'], ss['y'])

    def test_class_non_iid(self):
        splits: list[Split] = class_non_iid_split(
            self.iris['X'],
            self.iris['y'],
            n_splits=15,
            n_classes_per_split=2)

        self.assert_splits(splits, self.iris['y'], 15)
        self.assert_class_non_iid_splits(splits, 2)

        splits: list[Split] = class_non_iid_split(
            self.iris['X'],
            self.iris['y'],
            n_splits=17,
            n_classes_per_split=3
        )

        self.assert_splits(splits, self.iris['y'], 17)
        self.assert_class_non_iid_splits(splits, 3)

        splits: list[Split] = class_non_iid_split(
            self.digits['X'],
            self.digits['y'],
            n_splits=33,
            n_classes_per_split=4
        )

        self.assert_splits(splits, self.digits['y'], 33)
        self.assert_class_non_iid_splits(splits, 4)

        # With seeds
        splits: list[Split] = class_non_iid_split(
            self.digits['X'],
            self.digits['y'],
            n_splits=10,
            n_classes_per_split=4,
            seed=5
        )
        splits2: list[Split] = class_non_iid_split(
            self.digits['X'],
            self.digits['y'],
            n_splits=10,
            n_classes_per_split=4,
            seed=5
        )
        splits3: list[Split] = class_non_iid_split(
            self.digits['X'],
            self.digits['y'],
            n_splits=10,
            n_classes_per_split=4,
            seed=5
        )

        self.assert_same_splits(splits, splits2)
        self.assert_same_splits(splits, splits3)

    def test_dirichlet(self):
        splits: list[Split] = dirichlet_split(self.digits['X'],
                                              self.digits['y'],
                                              n_splits=50,
                                              alpha=0.5,
                                              split_min_size=1)

        self.assert_splits(splits, self.digits['y'], 50)
        self.assert_dirichlet_splits(splits, 1)

        splits: list[Split] = dirichlet_split(self.iris['X'],
                                              self.iris['y'],
                                              n_splits=12,
                                              alpha=0.5,
                                              split_min_size=2)

        self.assert_splits(splits, self.iris['y'], 12)
        self.assert_dirichlet_splits(splits, 2)

        # Binary case
        binary_y = self.iris['y']
        binary_y[binary_y == 2] = 1
        splits: list[Split] = dirichlet_split(self.iris['X'],
                                              binary_y,
                                              n_splits=12,
                                              alpha=0.5,
                                              split_min_size=2)

        self.assert_splits(splits, binary_y, 12)
        self.assert_dirichlet_splits(splits, 2)

        splits: list[Split] = dirichlet_split(self.iris['X'],
                                              self.iris['y'],
                                              n_splits=12,
                                              alpha=0.5,
                                              split_min_size=2,
                                              seed=544)
        splits2: list[Split] = dirichlet_split(self.iris['X'],
                                               self.iris['y'],
                                               n_splits=12,
                                               alpha=0.5,
                                               split_min_size=2,
                                               seed=544)
        splits3: list[Split] = dirichlet_split(self.iris['X'],
                                               self.iris['y'],
                                               n_splits=12,
                                               alpha=0.5,
                                               split_min_size=2,
                                               seed=544)

        self.assert_same_splits(splits, splits2)
        self.assert_same_splits(splits, splits3)
