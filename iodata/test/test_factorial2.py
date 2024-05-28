import pytest
import numpy as np
from ..overlap import factorial2


def test_integer_arguments():
    assert factorial2(0, exact=True) == 1
    assert factorial2(1, exact=True) == 1
    assert factorial2(2, exact=True) == 2
    assert factorial2(3, exact=True) == 3
    assert factorial2(-1, exact=True) == 1
    assert factorial2(-2, exact=True) == 0


def test_float_arguments():
    assert factorial2(0.0, exact=False) == pytest.approx(1.0)
    assert factorial2(1.0, exact=False) == pytest.approx(1.0)
    assert factorial2(2.0, exact=False) == pytest.approx(2.0)
    assert factorial2(3.0, exact=False) == pytest.approx(3.0)


def test_integer_array_argument():
    np.testing.assert_array_equal(
        factorial2(np.array([0, 1, 2, 3]), exact=True), np.array([1, 1, 2, 3])
    )


def test_float_array_argument():
    np.testing.assert_array_almost_equal(
        factorial2(np.array([0.0, 1.0, 2.0, 3.0]), exact=False), np.array([1.0, 1.0, 2.0, 3.0])
    )


def test_special_cases_exact():
    assert factorial2(-1, exact=True) == pytest.approx(1.0)
    assert factorial2(-2, exact=True) == pytest.approx(0.0)


def test_special_cases_not_exact():
    assert factorial2(-1.0, exact=False) == pytest.approx(1.0)
    assert factorial2(-2.0, exact=False) == pytest.approx(0.0)
