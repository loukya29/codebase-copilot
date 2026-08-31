import pytest
from calculator import add, subtract, multiply, divide, average


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 2) == 3


def test_multiply():
    assert multiply(4, 3) == 12


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero():
    # This test currently FAILS because divide() doesn't handle b == 0.
    # It's here on purpose so the agent's run_tests / suggest_fix tools
    # have something real to catch.
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)


def test_average_empty_list():
    # This test currently FAILS because average() doesn't handle an empty list.
    with pytest.raises(ValueError):
        average([])
