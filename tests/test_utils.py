import unittest
import utils

import numpy as np

class TestUtils(unittest.TestCase):
    def test_count(self):
        self.assertEqual(
            utils.count([1, 2, 3]),
            {1: 1, 2: 1, 3: 1}
        )

        self.assertEqual(
            utils.count([1, 1, 1, 1, 1]),
            {1: 5}
        )

        self.assertEqual(
            utils.count([]),
            {}
        )

        self.assertEqual(
            utils.count(("a", "e", "a", "b")),
            {"a": 2, "e": 1, "b": 1}
        )

    def test_top_n(self):
        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            utils.top_n(np.array([]), -1)

        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            utils.top_n(np.array([]), 2)

        self.assertEqual(utils.top_n(np.array([]), 0).tolist(), [])

        x = np.array([26, 5, 40, 19, 98, 31, 32, 37, 91, 20])

        self.assertEqual(utils.top_n(x, 1).tolist(), [98])
        self.assertEqual(utils.top_n(x, 5).tolist(), [98, 91, 40, 37, 32])

        x = np.array([1.1, 5.3, 0.6, 9.69, 1.12])
        self.assertEqual(utils.top_n(x, 1).tolist(), [9.69])
        self.assertEqual(utils.top_n(x, 5).tolist(),
                         [9.69, 5.3, 1.12, 1.1, 0.6])

    def test_bottom_n(self):
        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            utils.bottom_n(np.array([]), -1)

        with self.assertRaisesRegex(ValueError,
                                    'n should be 0 <= n <= x.shape'):
            utils.bottom_n(np.array([]), 2)

        self.assertEqual(utils.bottom_n(np.array([]), 0).tolist(), [])

        x = np.array([26, 5, 40, 19, 98, 31, 32, 37, 91, 20])

        self.assertEqual(utils.bottom_n(x, 1).tolist(), [5])
        self.assertEqual(utils.bottom_n(x, 5).tolist(), [5, 19, 20, 26, 31])

        x = np.array([1.1, 5.3, 0.6, 9.69, 1.12])
        self.assertEqual(utils.bottom_n(x, 1).tolist(), [0.6])
        self.assertEqual(utils.bottom_n(x, 5).tolist(),
                         [0.6, 1.1, 1.12, 5.3, 9.69])
