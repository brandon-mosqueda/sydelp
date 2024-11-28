from utils.utils import *

import numpy as np
from numpy import array
from numpy.testing import assert_array_equal, assert_array_almost_equal
from utils.typing import FloatArray

from unittest import TestCase


class TestUtils(TestCase):
    def test_count(self):
        self.assertEqual(count([1, 2, 3]), {1: 1, 2: 1, 3: 1})
        self.assertEqual(count([1, 1, 1, 1, 1]), {1: 5})
        self.assertEqual(count([]), {})
        self.assertEqual(count(("a", "e", "a", "b")), {"a": 2, "e": 1, "b": 1})

    def test_top_n(self):
        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            top_n(np.array([]), -1)

        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            top_n(np.array([]), 2)

        assert_array_equal(top_n(np.array([]), 0), [])

        x = np.array([26, 5, 40, 19, 98, 31, 32, 37, 91, 20])
        assert_array_equal(top_n(x, 1), [98])
        assert_array_equal(top_n(x, 5), [98, 91, 40, 37, 32])

        x = np.array([1.1, 5.3, 0.6, 9.69, 1.12])
        assert_array_equal(top_n(x, 1), [9.69])
        assert_array_equal(top_n(x, 5), [9.69, 5.3, 1.12, 1.1, 0.6])

    def test_bottom_n(self):
        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            bottom_n(np.array([]), -1)

        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            bottom_n(np.array([]), 2)

        assert_array_equal(bottom_n(np.array([]), 0), [])

        x = np.array([26, 5, 40, 19, 98, 31, 32, 37, 91, 20])
        assert_array_equal(bottom_n(x, 1), [5])
        assert_array_equal(bottom_n(x, 5), [5, 19, 20, 26, 31])

        x = np.array([1.1, 5.3, 0.6, 9.69, 1.12])
        assert_array_equal(bottom_n(x, 1), [0.6])
        assert_array_equal(bottom_n(x, 5), [0.6, 1.1, 1.12, 5.3, 9.69])

    def test_remove_indices(self):
        self.assertEqual(remove_indices([], []), [])
        self.assertEqual(remove_indices([], [1]), [])
        self.assertEqual(remove_indices([1], [1]), [1])

        self.assertEqual(remove_indices([1], [0]), [])
        self.assertEqual(remove_indices([1, 2, 3], [0, 2]), [2])
        self.assertEqual(remove_indices(['a', 'b', 'c'], [2]), ['a', 'b'])
        self.assertEqual(remove_indices(['a', 'b', 'c'], [2, 5]), ['a', 'b'])

    def assert_original_weights(self, x: list[FloatArray], y: list[FloatArray]):
        self.assertEqual(len(x), len(y), "x and y have not the same length")

        for i, j in zip(x, y):
            assert_array_almost_equal(i, j)

    def test_set_weights_to_array(self):
        arr: FloatArray = np.arange(3, dtype="float")
        weights: list[FloatArray] = [array([8, 9, 10])]

        set_weights_to_array(weights, arr)
        assert_array_equal(arr, array([8, 9, 10]))

        self.assert_original_weights(
            flat_weights_to_original(arr, weights_shapes=[(3,)]),
            weights
        )


        arr = np.arange(6, dtype="float")
        weights = [array([8, 9, 10]), array([93, 5, 8])]
        set_weights_to_array(weights, arr)
        assert_array_equal(arr, ([8, 9, 10, 93, 5, 8]))

        self.assert_original_weights(
            flat_weights_to_original(arr, weights_shapes=[(3,), (3,)]),
            weights
        )

        arr = np.arange(6, dtype="float")
        weights = [array([2]), array([3]), array([8]),
                   array([2]), array([3]), array([8])]
        set_weights_to_array(weights, arr)
        assert_array_equal(arr, ([2, 3, 8, 2, 3, 8]))

        self.assert_original_weights(
            flat_weights_to_original(arr, weights_shapes=[(1,), (1,), (1,),
                                                          (1,), (1,), (1,)]),
            weights
        )

        arr = np.arange(100, dtype="float")
        weights = [np.random.random((5, 5)),
                   np.random.random((5, 5)),
                   np.random.random((5, 5)),
                   np.random.random((5, 5)),]

        set_weights_to_array(weights, arr)
        assert_array_almost_equal(arr,np.concatenate([w.flatten()
                                                      for w in weights]))

        self.assert_original_weights(
            flat_weights_to_original(arr, weights_shapes=[(5, 5), (5, 5),
                                                          (5, 5), (5, 5),]),
            weights
        )
