import pytest
from m3util.util.kwargs import _filter_kwargs


def test_filter_kwargs_with_valid_args():
    def sample_function(a, b, c):
        pass

    kwargs = {"a": 1, "b": 2, "c": 3}
    filtered = _filter_kwargs(sample_function, kwargs)
    assert filtered == {"a": 1, "b": 2, "c": 3}


def test_filter_kwargs_with_invalid_args():
    def sample_function(a, b, c):
        pass

    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    filtered = _filter_kwargs(sample_function, kwargs)
    assert filtered == {"a": 1, "b": 2, "c": 3}


def test_filter_kwargs_with_no_args():
    def sample_function():
        pass

    kwargs = {"a": 1, "b": 2}
    filtered = _filter_kwargs(sample_function, kwargs)
    assert filtered == {}


def test_filter_kwargs_with_partial_valid_args():
    def sample_function(a, b):
        pass

    kwargs = {"a": 1, "b": 2, "c": 3}
    filtered = _filter_kwargs(sample_function, kwargs)
    assert filtered == {"a": 1, "b": 2}


def test_filter_kwargs_with_method():
    class SampleClass:
        def method(self, a, b):
            pass

    obj = SampleClass()
    kwargs = {"a": 1, "b": 2, "c": 3}
    filtered = _filter_kwargs(obj.method, kwargs)
    assert filtered == {"a": 1, "b": 2}
