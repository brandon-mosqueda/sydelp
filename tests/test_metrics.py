import numpy as np

from numpy import array
from numpy.typing import NDArray
from utils.metrics import *
from unittest import TestCase

class TestMetrics(TestCase):
    def test_label_flipping_success_rate(self):
        with self.assertRaisesRegex(
            ValueError,
            'y_true and y_pred must have the same length'):
            label_flipping_success_rate(array([]), array([1]), 1, 0)

        self.assertEqual(
            label_flipping_success_rate(array([]), array([]), 1, 0),
            0
        )

        self.assertEqual(
            label_flipping_success_rate(array([1]), array([1]), 1, 0),
            0
        )

        temp: NDArray[np.int64] = array([1, 1, 1, 0, 0, 0])

        self.assertEqual(label_flipping_success_rate(temp, temp, 1, 0), 0)
        self.assertEqual(label_flipping_success_rate(temp, temp[::-1], 1, 0), 1)

        # The changes in the other classes does not make a difference
        self.assertEqual(
            label_flipping_success_rate(temp, array([1, 1, 1, 1, 1, 1]), 1, 0),
            0
        )

        # One success flip
        self.assertEqual(
            label_flipping_success_rate(temp, array([0, 1, 1, 1, 1, 1]), 1, 0),
            1/3
        )
        self.assertEqual(
            label_flipping_success_rate(temp, array([0, 1, 0, 1, 1, 1]), 1, 0),
            2/3
        )

        self.assertEqual(
            label_flipping_success_rate(temp, array([0, 0, 0, 1, 1, 1]), 1, 0),
            1
        )

        x: NDArray = array([1, 2, 3, 4, 7, 0, 1, 3, 2, 7, 5, 7, 1, 7, 2, 7])
        # One flip
        y: NDArray = array([1, 2, 3, 4, 7, 0, 1, 3, 2, 7, 5, 7, 1, 5, 2, 7])
        # No flip to target class
        z: NDArray = array([1, 2, 3, 4, 0, 0, 1, 3, 2, 1, 5, 2, 1, 3, 2, 4])
        # Two success flips, other different flips
        a: NDArray = array([1, 2, 3, 4, 5, 0, 1, 3, 2, 3, 7, 5, 1, 2, 2, 1])
        # Random numbers one success flip
        b: NDArray = array([0, 5, 3, 7, 8, 3, 1, 0, 8, 5, 4, 2, 1, 3, 3, 7])
        # Full flips
        c: NDArray = array([0, 5, 3, 7, 5, 3, 1, 0, 8, 5, 4, 5, 1, 5, 3, 5])

        self.assertEqual(
            label_flipping_success_rate(x, y, 7, 5),
            1/5
        )

        self.assertEqual(
            label_flipping_success_rate(x, z, 7, 5),
            0
        )

        self.assertEqual(
            label_flipping_success_rate(x, a, 7, 5),
            2/5
        )

        self.assertEqual(
            label_flipping_success_rate(x, b, 7, 5),
            1/5
        )

        self.assertEqual(
            label_flipping_success_rate(x, c, 7, 5),
            1
        )

        # Test with mixed source labels and no flips to target
        self.assertEqual(
            label_flipping_success_rate(
                array([1, 1, 2, 2, 1]),
                array([2, 2, 2, 2, 2]),
                1,
                0
            ),
            0
        )

        # Test with large arrays
        large_y_true = np.random.randint(0, 10, 1000)
        large_y_pred = np.random.randint(11, 20, 1000)
        self.assertEqual(
            label_flipping_success_rate(large_y_true, large_y_pred, 5, 3),
            0
        )

    def test_label_recall(self):
        with self.assertRaisesRegex(
                ValueError,
                'y_true and y_pred must have the same length'):
            label_recall(array([]), array([1]), 1)

        self.assertEqual(
            label_recall(array([]), array([]), 1),
            0
        )

        self.assertEqual(
            label_recall(array([0]), array([1]), 1),
            0
        )

        # Different class than desired
        self.assertEqual(
            label_recall(array([5]), array([5]), 1),
            0
        )

        self.assertEqual(
            label_recall(array([5]), array([5]), 5),
            1
        )

        self.assertEqual(
            label_recall(array([5, 5, 5]), array([5, 5, 5]), 5),
            1
        )

        self.assertEqual(
            label_recall(array([0, 1, 2, 3]), array([3, 2, 1, 0]), 3),
            0
        )

        self.assertEqual(
            label_recall(array([0, 1, 2, 3]), array([3, 2, 1, 3]), 3),
            1
        )

        x: NDArray = array([0, 0, 0, 1, 1, 1])
        # One error
        y: NDArray = array([1, 1, 1, 1, 1, 0])
        self.assertEqual(label_recall(x, y, 1), 2/3)

        # Two errors
        y: NDArray = array([0, 0, 0, 0, 1, 0])
        self.assertEqual(label_recall(x, y, 1), 1/3)

        x = array([0, 0, 0, 0, 10, 5, 1, 0, 0, 0, 0])
        y = array([1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1])
        self.assertEqual(label_recall(x, y, 10), 1)

        y = array([1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1])
        self.assertEqual(label_recall(x, y, 10), 0)

        # Test with multiple labels
        x = array([0, 1, 2, 3, 4, 5])
        y = array([0, 1, 2, 3, 4, 5])
        self.assertEqual(label_recall(x, y, 0), 1)
        self.assertEqual(label_recall(x, y, 1), 1)
        self.assertEqual(label_recall(x, y, 2), 1)
        self.assertEqual(label_recall(x, y, 3), 1)
        self.assertEqual(label_recall(x, y, 4), 1)
        self.assertEqual(label_recall(x, y, 5), 1)

        # Test with some incorrect predictions
        y = array([0, 1, 0, 3, 4, 0])
        self.assertEqual(label_recall(x, y, 0), 1)
        self.assertEqual(label_recall(x, y, 1), 1)
        self.assertEqual(label_recall(x, y, 2), 0)
        self.assertEqual(label_recall(x, y, 3), 1)
        self.assertEqual(label_recall(x, y, 4), 1)
        self.assertEqual(label_recall(x, y, 5), 0)

        # Test with no occurrences of the label in y_true
        x = array([0, 0, 0, 0])
        y = array([1, 1, 1, 1])
        self.assertEqual(label_recall(x, y, 1), 0)

        # Test with all incorrect predictions
        x = array([0, 1, 2, 3])
        y = array([3, 2, 1, 0])
        self.assertEqual(label_recall(x, y, 0), 0)
        self.assertEqual(label_recall(x, y, 1), 0)
        self.assertEqual(label_recall(x, y, 2), 0)
        self.assertEqual(label_recall(x, y, 3), 0)

        # Test with mixed correct and incorrect predictions
        x = array([0, 1, 2, 3, 4, 5])
        y = array([0, 2, 2, 3, 5, 5])
        self.assertEqual(label_recall(x, y, 0), 1)
        self.assertEqual(label_recall(x, y, 1), 0)
        self.assertEqual(label_recall(x, y, 2), 1)
        self.assertEqual(label_recall(x, y, 3), 1)
        self.assertEqual(label_recall(x, y, 4), 0)
        self.assertEqual(label_recall(x, y, 5), 1)

        # No label at all
        x = array([0, 1, 2, 3, 4, 5])
        y = array([0, 2, 2, 3, 5, 5])
        self.assertEqual(label_recall(x, y, 15), 0)

        # Test with large arrays
        large_y_true = np.random.randint(0, 10, 1000)
        large_y_pred = np.random.randint(11, 20, 1000)
        self.assertEqual(
            label_recall(large_y_true, large_y_pred, 5),
            0
        )
