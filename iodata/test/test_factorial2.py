import numpy as np
import pytest

from ..overlap import factorial2


def test_integer_arguments():
    assert factorial2(0) == 1
    assert factorial2(1) == 1
    assert factorial2(2) == 2
    assert factorial2(3) == 3
    assert factorial2(-1) == 1
    assert factorial2(-2) == 0


def test_float_arguments():
    with pytest.raises(TypeError):
        assert factorial2(1.0)


def test_integer_array_argument():
    assert (factorial2(np.array([0, 1, 2, 3])) == np.array([1, 1, 2, 3])).all()


def test_float_array_argument():
    with pytest.raises(TypeError):
        factorial2(np.array([0.0, 1.0, 2.0, 3.0]))


def test_special_cases_exact():
    assert factorial2(-1) == pytest.approx(1)
    assert factorial2(-2) == pytest.approx(0)
