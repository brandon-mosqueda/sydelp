import numpy as np

from aggregation import *
from numpy.testing import assert_array_equal
from initialize import iris_model
from unittest import TestCase


class TestAggregation(TestCase):
    def test_weights_distance(self):
        self.assertEqual(weights_distance([], []), 0)
        x = [np.array([1.0])]
        y = [np.array([4.0])]
        self.assertEqual(weights_distance(x, y), 3)

        x = np.array([1, 2, 3])
        y = np.array([2, 3, 4])

        self.assertEqual(weights_distance([x], [x]), 0)
        self.assertEqual(weights_distance([y], [y]), 0)

        self.assertAlmostEqual(weights_distance([x], [y]), np.sqrt(3), places=6)
        self.assertAlmostEqual(
            weights_distance([x, y], [y, x]),
            np.sqrt(6),
            places=6
        )

        # Test case with very large values in weights
        x = [np.array([1e10, 2e10])]
        y = [np.array([3e10, 4e10])]
        self.assertAlmostEqual(
            weights_distance(x, y),
            np.sqrt((2e10)**2 + (2e10)**2, dtype="float128"),
            places=6
        )

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
            np.sqrt(expected_result),
            places=6
        )
