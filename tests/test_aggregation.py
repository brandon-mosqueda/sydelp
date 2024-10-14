import numpy as np

from aggregation import *
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase


class TestAggregation(TestCase):
    def test_weights_distances(self):
        # Zero distance and empty weights
        self.assertEqual(weights_distance([], []), 0)
        assert_array_equal(
            weights_distance_matrix([]),
            np.zeros((0, 0), dtype="float128"),
            strict=True
        )
        assert_array_equal(
            weights_distance_matrix([[]]),
            np.array([[0]], dtype="float128"),
            strict=True
        )
        assert_array_equal(
            weights_distance_matrix([[np.arange(10)]]),
            np.array([[0]], dtype="float128"),
            strict=True
        )

        # Single element distance
        x = [np.array([1.0])]
        y = [np.array([4.0])]
        self.assertEqual(weights_distance(x, y), 3)
        assert_array_equal(
            weights_distance_matrix([x, y]),
            np.array([[0, 3], [3, 0]], dtype="float128"),
            strict=True
        )

        # Multiple element distance
        x = np.array([1, 2, 3])
        y = np.array([2, 3, 4])
        self.assertEqual(weights_distance([x], [x]), 0)
        self.assertEqual(weights_distance([y], [y]), 0)
        self.assertAlmostEqual(weights_distance([x], [y]), np.sqrt(3))
        self.assertAlmostEqual(
            weights_distance([x, y], [y, x]),
            np.sqrt(6)
        )
        assert_array_equal(
            weights_distance_matrix([[x], [x]]),
            np.array([[0, 0], [0, 0]], dtype="float128"),
            strict=True
        )
        assert_array_almost_equal(
            weights_distance_matrix([[x], [y]]),
            [[0, np.sqrt(3)], [np.sqrt(3), 0]]
        )
        assert_array_almost_equal(
            weights_distance_matrix([[x, y], [y, x]]),
            [[0, np.sqrt(6)], [np.sqrt(6), 0]]
        )

        # Large numbers
        x = [np.array([1e20, 2e20])]
        y = [np.array([3e20, 4e20])]
        expected_result = np.sqrt((2e20)**2 + (2e20)**2, dtype="float128")
        self.assertAlmostEqual(
            weights_distance(x, y),
            expected_result
        )
        assert_array_almost_equal(
            weights_distance_matrix([x, y]),
            [[0, expected_result], [expected_result, 0]]
        )

        # Complex weights
        x = [
            np.array([0.08, 0.72, 0.67]),
            np.array([0.01, 0.53, 0.24]),
            np.array([[0.42, 0.42], [0.35, 0.64]])
        ]
        y = [
            np.array([0.82, 0.88, 0.02]),
            np.array([0.27, 0.37, 0.11]),
            np.array([[0.84, 0.76], [0.02, 0.34]])
        ]
        expected_result = np.sum([
            (0.08 - 0.82)**2, (0.72 - 0.88)**2, (0.67 - 0.02)**2,
            (0.01 - 0.27)**2, (0.53 - 0.37)**2, (0.24 - 0.11)**2,
            (0.42 - 0.84)**2, (0.42 - 0.76)**2,
            (0.35 - 0.02)**2, (0.64 - 0.34)**2
        ], dtype="float128")
        self.assertAlmostEqual(
            weights_distance(x, y),
            np.sqrt(expected_result)
        )
        assert_array_almost_equal(
            weights_distance_matrix([x, y]),
            [[0, np.sqrt(expected_result)], [np.sqrt(expected_result), 0]]
        )
