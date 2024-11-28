import numpy as np

from numpy import array
from utils.aggregation import *
from utils.typing import FloatArray
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase


class TestAggregation(TestCase):
    x: FloatArray = array([
        0.08, 0.72, 0.67,
        0.01, 0.53, 0.24,
        0.42, 0.42,
        0.35, 0.64
    ])
    y: FloatArray = array([
        0.82, 0.88, 0.02,
        0.27, 0.37, 0.11,
        0.84, 0.76,
        0.02, 0.34
    ])
    z: FloatArray = array([
        0.45, 0.30, 0.55,
        0.21, 0.43, 0.17,
        0.12, 0.45,
        0.33, 0.21
    ])

    def test_fed_avg(self):
        # Empty params
        assert_array_equal(fed_avg(array([[]])), np.zeros((1, 0)), strict=True)
        assert_array_equal(fed_avg(array([[], []])),
                           np.zeros((1, 0)),
                           strict=True)
        assert_array_equal(fed_avg(array([[], [], []])),
                           np.zeros((1, 0)),
                           strict=True)

        # Single element
        assert_array_equal(fed_avg(array([[5]])),
                           array([5.0]),
                           strict=True)
        assert_array_equal(fed_avg(array([[1, 2, 3]])),
                           array([1.0, 2.0, 3.0]),
                           strict=True)

        # Multiple elements
        assert_array_equal(
            fed_avg(array([[1, 2],
                           [3, 4]])),
            array([2.0, 3.0]),
            strict=True
        )
        assert_array_equal(
            fed_avg(array([[1, 2],
                           [3, 4],
                           [5, 6]])),
            array([3.0, 4.0]),
            strict=True
        )
        assert_array_equal(
            fed_avg(array([[1, 2, 3],
                           [4, 5, 6]])),
            array([2.5, 3.5, 4.5]),
            strict=True
        )

        # Complex params
        expected_result = array([
            (0.08 + 0.82 + 0.45) / 3,
                (0.72 + 0.88 + 0.30) / 3,
                (0.67 + 0.02 + 0.55) / 3,
            (0.01 + 0.27 + 0.21) / 3,
                (0.53 + 0.37 + 0.43) / 3,
                (0.24 + 0.11 + 0.17) / 3,
            (0.42 + 0.84 + 0.12) / 3, (0.42 + 0.76 + 0.45) / 3,
                (0.35 + 0.02 + 0.33) / 3, (0.64 + 0.34 + 0.21) / 3
        ])
        assert_array_almost_equal(
            fed_avg(array([self.x, self.y, self.z])),
            expected_result
        )
        # Different order does not change the result
        assert_array_almost_equal(
            fed_avg(array([self.z, self.x, self.y])),
            expected_result
        )

    def test_krum(self):
        # Empty params
        assert_array_equal(krum(array([[]]), 0), np.zeros((1, 0)), strict=True)
        assert_array_equal(krum(array([[], [], [], []]), 2),
                           np.zeros((1, 0)),
                           strict=True)

        # Single element
        assert_array_equal(krum(array([[5]]), 0),
                           array([5.0]),
                           strict=True)
        assert_array_equal(krum(array([[1, 2, 3]]), 0),
                           array([1.0, 2.0, 3.0]),
                           strict=True)

        # Multiple elements
        assert_array_equal(
            krum(array([[1, 2],
                        [3, 4]]),
                 0),
            array([2.0, 3.0]),
            strict=True
        )

        # Multiple elements with m
        params = array([[1, 2],
                        [3, 4],
                        [15, 3],
                        [10, 2]])
        #         1       2      3      4
        # 1  0.0000  2.8284 9.0000 14.036
        # 2  2.8284  0.0000 7.2801 12.042
        # 3  9.0000  7.2801 0.0000  5.099
        # 4 14.0357 12.0416 5.0990  0.000
        # 1, 2 have 3 the lowest score. In fact 3 and 4 have the same (5.099)
        # because only one element is left on n_models = n - m - 2. But as 3
        # appears first, it will be the one taken.
        assert_array_equal(
            krum(params, 1),
            array([1.0, 2.0]),
            strict=True
        )
        # 1 and 2 are selected
        assert_array_equal(
            krum(params, 0),
            array([2.0, 3.0]),
            strict=True
        )

        # Complex params
        w = array([
            5.37, 3.85, 2.50,
            5.24, 8.52, 2.29,
            4.00, 2.39,
            8.55, 6.26
        ])
        #         x      w       y        z
        # x  0.00000 15.876  1.2636  0.81166
        # w 15.87599  0.000 15.8745 16.08862
        # y  1.26361 15.874  0.0000  1.22168
        # z  0.81166 16.089  1.2217  0.00000
        # Total sums (this value does not always reflect the KRUM score):
        #      x      w      y      z
        # 17.951 47.839 18.360 18.122
        assert_array_almost_equal(krum(array([self.x, w, self.y, self.z]), 0),
                                  (self.x + self.z) / 2)
        # Different order
        assert_array_almost_equal(krum(array([self.z, w, self.x, self.y]), 1),
                                  self.z)

        # All the random params will be excluded
        assert_array_almost_equal(
            krum(array([self.x,
                        np.random.uniform(0, 100, self.x.size),
                        w,
                        np.random.uniform(0, 100, self.x.size),
                        self.y,
                        np.random.uniform(0, 100, self.x.size),
                        np.random.uniform(0, 100, self.x.size),
                        self.z]),
                 2),
            # The 4 random vectors are discarded when m = 2
            (self.x + self.z + w + self.y) / 4
        )

    def test_fed_avg_with_different_weights(self):
        # Different weights
        assert_array_equal(
            fed_avg(array([[1, 2],
                           [3, 4]]),
                    array([0.25, 0.75])),
            array([0.25 + 3*0.75, 2*0.25 + 4*0.75]),
            strict=True
        )

        assert_array_equal(
            fed_avg(array([[1, 2],
                           [3, 4],
                           [5, 6]]),
                    array([0.25, 0.5, 0.25])),
            array([0.25 + 3*0.5 + 5*0.25,
                   2*0.25 + 4*0.5 + 6*0.25]),
            strict=True
        )

        assert_array_equal(
            fed_avg(array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]),
                    array([1/6, 2/6, 3/6])),
            array([1/6 + 2/6*4 + 3/6*7,
                   1/6*2 + 2/6*5 + 3/6*8,
                   1/6*3 + 2/6*6 + 3/6*9]),
            strict=True
        )

        # More model_params of length 3
        params = array([
            [6, 8, 7, 2, 5, 9, 6, 7, 7],
            [1, 7, 7, 7, 3, 6, 9, 3, 5],
            [6, 1, 9, 1, 2, 5, 9, 6, 8]
        ])
        w = array([7, 2, 4]) / (7 + 2 + 4)

        expected_result = array(
            params[0] * w[0] +
            params[1] * w[1] +
            params[2] * w[2]
        )
        assert_array_equal(
            fed_avg(params, w),
            expected_result,
            strict=True
        )
