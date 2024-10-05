import torch
import numpy as np
import pytest
from m3util.converters.complex import to_complex


# Test for valid PyTorch tensor input
def test_to_complex_valid_tensor():
    data = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)]
    result = to_complex(data)
    assert torch.is_complex(result), "Tensor should be converted to a complex type"
    assert result.dtype == torch.complex64, "Tensor dtype should be torch.complex64"


# Test for already complex PyTorch tensor input
def test_to_complex_already_complex_tensor():
    data = [torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex64)]
    result = to_complex(data)
    assert torch.is_complex(result), "Tensor should remain a complex type"
    assert (
        result.dtype == torch.complex64
    ), "Tensor dtype should still be torch.complex64"


# Test for valid NumPy array input
def test_to_complex_valid_numpy():
    data = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    result = to_complex(data)
    assert np.iscomplexobj(result), "NumPy array should be converted to a complex type"
    assert result.dtype == np.complex64, "NumPy array dtype should be np.complex64"


# Test for already complex NumPy array input
def test_to_complex_already_complex_numpy():
    data = [np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)]
    result = to_complex(data)
    assert np.iscomplexobj(result), "NumPy array should remain a complex type"
    assert (
        result.dtype == np.complex64
    ), "NumPy array dtype should still be np.complex64"

# Test for empty input (should raise IndexError)
def test_to_complex_empty_input():
    data = []
    with pytest.raises(
        IndexError, match="Input data is empty or not an array-like structure."
    ):
        to_complex(data)


# Test for input that is not array-like (should raise IndexError)
def test_to_complex_non_array_like_input():
    data = 10
    with pytest.raises(
        IndexError, match="Input data is empty or not an array-like structure."
    ):
        to_complex(data)


# Test for invalid internal structure leading to a runtime error
def test_to_complex_runtime_error():
    class CustomObj:
        pass

    data = [CustomObj()]
    with pytest.raises(
        RuntimeError, match="An error occurred while processing the data: .*"
    ):
        to_complex(data)
