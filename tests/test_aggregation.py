import numpy as np

from aggregation import *
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase


class TestAggregation(TestCase):
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
    z = [
        np.array([0.45, 0.30, 0.55]),
        np.array([0.21, 0.43, 0.17]),
        np.array([[0.12, 0.45], [0.33, 0.21]])
    ]

    @staticmethod
    def random_weights(ref: Weights,
                       low: Union[float, int] = 0,
                       high: Union[float, int] = 1) -> Weights:
        return [np.random.uniform(low, high, arr.shape) for arr in ref]

    def assert_weights(self, x, y):
        self.assertTrue(isinstance(x, list), "x is not a list")
        self.assertTrue(isinstance(y, list), "y is not a list")
        self.assertEqual(len(x), len(y), "x and y have different lengths")

        for x_i, y_i in zip(x, y):
            self.assertIsInstance(x_i, np.ndarray, "x_i is not a numpy array")
            self.assertIsInstance(y_i, np.ndarray, "y_i is not a numpy array")
            self.assertEqual(x_i.shape,
                             y_i.shape,
                             "x_i and y_i have different shapes")
            self.assertEqual(x_i.dtype,
                             y_i.dtype,
                             "x_i and y_i have different dtypes")

            assert_array_almost_equal(x_i, y_i, err_msg="x_i and y_i differ")

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
        expected_result = np.sum([
            (0.08 - 0.82)**2, (0.72 - 0.88)**2, (0.67 - 0.02)**2,
            (0.01 - 0.27)**2, (0.53 - 0.37)**2, (0.24 - 0.11)**2,
            (0.42 - 0.84)**2, (0.42 - 0.76)**2,
            (0.35 - 0.02)**2, (0.64 - 0.34)**2
        ], dtype="float128")
        self.assertAlmostEqual(
            weights_distance(self.x, self.y),
            np.sqrt(expected_result)
        )
        assert_array_almost_equal(
            weights_distance_matrix([self.x, self.y]),
            [[0, np.sqrt(expected_result)], [np.sqrt(expected_result), 0]]
        )

    def test_fed_avg(self):
        # Empty weights
        self.assertEqual(fed_avg([]), [])
        self.assertEqual(fed_avg([[], []]), [])

        # Single element
        assert_array_equal(fed_avg([[np.array([5])]]),
                           [np.array([5.0])],
                           strict=True)
        assert_array_equal(fed_avg([[np.array([1, 2, 3])]]),
                           [np.array([1.0, 2.0, 3.0])],
                           strict=True)

        # Multiple elements
        assert_array_equal(
            fed_avg([[np.array([1, 2])],
                     [np.array([3, 4])]]),
            [np.array([2.0, 3.0])],
            strict=True
        )
        assert_array_equal(
            fed_avg([[np.array([1, 2])],
                     [np.array([3, 4])],
                     [np.array([5, 6])]]),
            [np.array([3.0, 4.0])],
            strict=True
        )
        assert_array_equal(
            fed_avg([[np.array([1, 2, 3])],
                     [np.array([4, 5, 6])]]),
            [np.array([2.5, 3.5, 4.5])],
            strict=True
        )

        # Complex weights
        expected_result = [
            np.array([(0.08 + 0.82 + 0.45) / 3,
                      (0.72 + 0.88 + 0.30) / 3,
                      (0.67 + 0.02 + 0.55) / 3]),
            np.array([(0.01 + 0.27 + 0.21) / 3,
                      (0.53 + 0.37 + 0.43) / 3,
                      (0.24 + 0.11 + 0.17) / 3]),
            np.array([[(0.42 + 0.84 + 0.12) / 3, (0.42 + 0.76 + 0.45) / 3],
                      [(0.35 + 0.02 + 0.33) / 3, (0.64 + 0.34 + 0.21) / 3]])
        ]
        self.assert_weights(fed_avg([self.x, self.y, self.z]), expected_result)
        # Different order does not change the result
        self.assert_weights(fed_avg([self.z, self.x, self.y]), expected_result)

    def test_krum(self):
        # Empty weights
        self.assertEqual(krum([], 0), [])
        self.assertEqual(krum([[], [], [], []], 2), [])

        # Single element
        assert_array_equal(krum([[np.array([5])]], 0),
                           [np.array([5.0])],
                           strict=True)
        assert_array_equal(krum([[np.array([1, 2, 3])]], 0),
                           [np.array([1.0, 2.0, 3.0])],
                           strict=True)

        # Multiple elements
        assert_array_equal(
            krum([[np.array([1, 2])],
                  [np.array([3, 4])]], 0),
            [np.array([2.0, 3.0])],
            strict=True
        )

        # Multiple elements with m
        weights = [[np.array([1, 2])],
                   [np.array([3, 4])],
                   [np.array([15, 3])],
                   [np.array([10, 2])]]
        #         1       2      3      4
        # 1  0.0000  2.8284 9.0000 14.036
        # 2  2.8284  0.0000 7.2801 12.042
        # 3  9.0000  7.2801 0.0000  5.099
        # 4 14.0357 12.0416 5.0990  0.000
        # 1, 2 have 3 the lowest score. In fact 3 and 4 have the same (5.099)
        # because only one element is left on n_models = n - m - 2. But as 3
        # appears first, it will be the one taken.
        assert_array_equal(
            krum(weights, 1),
            [np.array([19/3, 9/3])],
            strict=True
        )
        assert_array_equal(
            krum(weights, 2),
            [np.array([2.0, 3.0])],
            strict=True
        )
        # m = 3 is the same as m = 2 = min(int(m), n - 2)
        assert_array_equal(
            krum(weights, 3),
            [np.array([2.0, 3.0])],
            strict=True
        )

        # Complex weights
        w = [
            np.array([5.37, 3.85, 2.50]),
            np.array([5.24, 8.52, 2.29]),
            np.array([[4.00, 2.39], [8.55, 6.26]])
        ]
        #         x      w       y        z
        # x  0.00000 15.876  1.2636  0.81166
        # w 15.87599  0.000 15.8745 16.08862
        # y  1.26361 15.874  0.0000  1.22168
        # z  0.81166 16.089  1.2217  0.00000
        # Total sums (this value does not always reflect the KRUM score):
        #      x      w      y      z
        # 17.951 47.839 18.360 18.122
        expected_result_all = [
            np.array([(0.08 + 0.82 + 0.45 + 5.37) / 4,
                      (0.72 + 0.88 + 0.30 + 3.85) / 4,
                      (0.67 + 0.02 + 0.55 + 2.50) / 4]),
            np.array([(0.01 + 0.27 + 0.21 + 5.24) / 4,
                      (0.53 + 0.37 + 0.43 + 8.52) / 4,
                      (0.24 + 0.11 + 0.17 + 2.29) / 4]),
            np.array([[(0.42 + 0.84 + 0.12 + 4.0) / 4,
                       (0.42 + 0.76 + 0.45 + 2.39) / 4],
                      [(0.35 + 0.02 + 0.33 + 8.55) / 4,
                       (0.64 + 0.34 + 0.21 + 6.26) / 4]])
        ]
        self.assert_weights(krum([self.x, w, self.y, self.z], 0),
                            expected_result_all)
        # The same as FedAvg
        self.assert_weights(krum([self.x, w, self.y, self.z], 0),
                            fed_avg([self.x, w, self.y, self.z]))

        expected_result = [
            np.array([(0.08 + 0.82 + 0.45) / 3,
                      (0.72 + 0.88 + 0.30) / 3,
                      (0.67 + 0.02 + 0.55) / 3]),
            np.array([(0.01 + 0.27 + 0.21) / 3,
                      (0.53 + 0.37 + 0.43) / 3,
                      (0.24 + 0.11 + 0.17) / 3]),
            np.array([[(0.42 + 0.84 + 0.12) / 3, (0.42 + 0.76 + 0.45) / 3],
                      [(0.35 + 0.02 + 0.33) / 3, (0.64 + 0.34 + 0.21) / 3]])
        ]
        # w will be excluded from the final result
        self.assert_weights(krum([self.x, w, self.y, self.z], 1),
                            expected_result)
        # Different order does not change the result
        self.assert_weights(krum([w, self.x, self.z, self.y], 1),
                            expected_result)

        # expected_result is x and w
        expected_result = [
            np.array([(0.08 + 5.37) / 2, (0.72 + 3.85) / 2, (0.67 + 2.50) / 2]),
            np.array([(0.01 + 5.24) / 2, (0.53 + 8.52) / 2, (0.24 + 2.29) / 2]),
            np.array([[(0.42 + 4.00) / 2, (0.42 + 2.39) / 2],
                      [(0.35 + 8.55) / 2, (0.64 + 6.26) / 2]])
        ]
        # with m = 2, the score is 0 for everyone, so the result will be the
        # average of x and w that are the first two elements
        self.assert_weights(krum([self.x, w, self.y, self.z], 2),
                            expected_result)

        # All the random weights will be excluded
        self.assert_weights(
            krum([self.x,
                  self.random_weights(self.x, 0, 100),
                  w,
                  self.random_weights(self.x, 0, 100),
                  self.y,
                  self.random_weights(self.x, 0, 100),
                  self.random_weights(self.x, 0, 100),
                  self.z], 4),
            expected_result_all
        )
