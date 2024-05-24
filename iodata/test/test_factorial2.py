import unittest
import numpy as np
from .. import overlap

factorial2 = overlap.factorial2


class TestFactorial2(unittest.TestCase):
    def test_integer_arguments(self):
        self.assertEqual(factorial2(0, exact=True), 1)
        self.assertEqual(factorial2(1, exact=True), 1)
        self.assertEqual(factorial2(2, exact=True), 2)
        self.assertEqual(factorial2(3, exact=True), 3)
        self.assertEqual(factorial2(-1, exact=True), 1.0)
        self.assertEqual(factorial2(-2, exact=True), 0.0)

    def test_float_arguments(self):
        np.testing.assert_almost_equal(factorial2(0.0, exact=False), 1.0)
        np.testing.assert_almost_equal(factorial2(1.0, exact=False), 1.0)
        np.testing.assert_almost_equal(factorial2(2.0, exact=False), 2.0)
        np.testing.assert_almost_equal(factorial2(3.0, exact=False), 3.0)

    def test_integer_array_argument(self):
        np.testing.assert_array_equal(
            factorial2(np.array([0, 1, 2, 3]), exact=True), np.array([1, 1, 2, 3])
        )

    def test_float_array_argument(self):
        np.testing.assert_array_almost_equal(
            factorial2(np.array([0.0, 1.0, 2.0, 3.0]), exact=False), np.array([1.0, 1.0, 2.0, 3.0])
        )

    def test_special_cases_exact(self):
        self.assertEqual(factorial2(-1, exact=True), 1.0)
        self.assertEqual(factorial2(-2, exact=True), 0.0)

    def test_special_cases_not_exact(self):
        np.testing.assert_almost_equal(factorial2(-1.0, exact=False), 1.0)
        np.testing.assert_almost_equal(factorial2(-2.0, exact=False), 0.0)


if __name__ == "__main__":
    unittest.main()
