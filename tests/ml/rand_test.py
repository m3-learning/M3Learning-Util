import torch
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Skip TensorFlow tests if it's not installed
try:
    import tensorflow as tf

    tensorflow_available = True
except ImportError:
    tensorflow_available = False

from m3util.ml.rand import (
    rand_tensor,
    set_seeds,
)  # Assuming your functions are in 'your_module.py'


@pytest.mark.parametrize(
    "min_val, max_val, size",
    [
        (0, 1, (2, 2)),
        (-5, 5, (3, 4)),
        (10, 100, (1,)),
    ],
)
def test_rand_tensor_range(min_val, max_val, size):
    # Generate a tensor
    tensor = rand_tensor(min=min_val, max=max_val, size=size)

    # Check the shape of the tensor
    assert tensor.shape == torch.Size(
        size
    ), f"Tensor shape should be {size}, but got {tensor.shape}"

    # Check that values are within the specified range
    assert (tensor >= min_val).all() and (
        tensor <= max_val
    ).all(), f"Values should be in range [{min_val}, {max_val}]"


def test_rand_tensor_reproducibility():
    # Set seed and generate two tensors
    set_seeds(42, pytorch_=True, numpy_=False, tensorflow_=False)
    tensor1 = rand_tensor(min=0, max=1, size=(5,))

    set_seeds(42, pytorch_=True, numpy_=False, tensorflow_=False)
    tensor2 = rand_tensor(min=0, max=1, size=(5,))

    # Check that the tensors are equal
    assert torch.equal(
        tensor1, tensor2
    ), "Tensors should be equal when the same seed is set."


@pytest.mark.parametrize(
    "seed, pytorch_, numpy_, tensorflow_",
    [
        (42, True, True, True),
        (1234, True, True, False),
        (999, True, False, False),
    ],
)
def test_set_seeds(seed, pytorch_, numpy_, tensorflow_):
    # Set the seeds
    set_seeds(seed, pytorch_=pytorch_, numpy_=numpy_, tensorflow_=tensorflow_)

    # Check PyTorch seed
    if pytorch_:
        tensor1 = rand_tensor(min=0, max=1, size=(5,))
        set_seeds(seed, pytorch_=True, numpy_=False, tensorflow_=False)
        tensor2 = rand_tensor(min=0, max=1, size=(5,))
        assert torch.equal(
            tensor1, tensor2
        ), f"PyTorch should return the same random tensor with seed {seed}."

    # Check NumPy seed
    if numpy_:
        np1 = np.random.rand()
        set_seeds(seed, pytorch_=False, numpy_=True, tensorflow_=False)
        np2 = np.random.rand()
        assert (
            np1 == np2
        ), f"NumPy should return the same random number with seed {seed}."

    # Check TensorFlow seed (only if TensorFlow is available)
    if tensorflow_ and tensorflow_available:
        tf1 = tf.random.uniform((1,)).numpy()[0]
        set_seeds(seed, pytorch_=False, numpy_=False, tensorflow_=True)
        tf2 = tf.random.uniform((1,)).numpy()[0]
        assert (
            tf1 == tf2
        ), f"TensorFlow should return the same random number with seed {seed}."


@pytest.mark.skipif(not tensorflow_available, reason="TensorFlow not installed")
def test_set_seeds_tensorflow():
    seed = 42
    set_seeds(seed, pytorch_=False, numpy_=False, tensorflow_=True)

    tf1 = tf.random.uniform((1,)).numpy()[0]
    set_seeds(seed, pytorch_=False, numpy_=False, tensorflow_=True)
    tf2 = tf.random.uniform((1,)).numpy()[0]

    assert (
        tf1 == tf2
    ), "TensorFlow should return the same random number with the same seed."


def test_set_seeds_pytorch():
    seed = 123
    set_seeds(seed, pytorch_=True, numpy_=False, tensorflow_=False)

    tensor1 = rand_tensor(min=0, max=1, size=(5,))
    set_seeds(seed, pytorch_=True, numpy_=False, tensorflow_=False)
    tensor2 = rand_tensor(min=0, max=1, size=(5,))

    assert torch.equal(
        tensor1, tensor2
    ), "PyTorch should return the same random tensor with the same seed."


def test_set_seeds_numpy():
    seed = 123
    set_seeds(seed, pytorch_=False, numpy_=True, tensorflow_=False)

    np1 = np.random.rand()
    set_seeds(seed, pytorch_=False, numpy_=True, tensorflow_=False)
    np2 = np.random.rand()

    assert np1 == np2, "NumPy should return the same random number with the same seed."


@pytest.mark.skipif(not tensorflow_available, reason="TensorFlow not installed")
def test_set_seeds_tensorflow__():
    seed = 123
    set_seeds(seed, pytorch_=False, numpy_=False, tensorflow_=True)

    tf1 = tf.random.uniform((1,)).numpy()[0]
    set_seeds(seed, pytorch_=False, numpy_=False, tensorflow_=True)
    tf2 = tf.random.uniform((1,)).numpy()[0]

    assert (
        tf1 == tf2
    ), "TensorFlow should return the same random number with the same seed."
    
### Test for rand_tensor function ###

@patch("builtins.print")
@patch("torch.rand", side_effect=ImportError("torch not found"))
def test_rand_tensor_torch_not_installed(mock_torch_rand, mock_print):
    """Test that the function raises an error if torch is not installed"""
    with pytest.raises(ImportError):
        rand_tensor(min=0, max=1, size=(1))


### Test for set_seeds function ###


@patch("numpy.random.seed")
@patch("builtins.print")
def test_set_seeds_numpy_(mock_print, mock_numpy_seed):
    """Test that Numpy seed is set correctly"""
    set_seeds(seed=123, numpy_=True)



@patch("tensorflow.random.set_seed")
@patch("builtins.print")
def test_set_seeds_tensorflow_(mock_print, mock_tf_seed):
    """Test that TensorFlow seed is set correctly"""
    set_seeds(seed=123, tensorflow_=True)

    # Check if TensorFlow seed-setting function was called
    mock_tf_seed.assert_called_once_with(123)



@patch("builtins.print")
@patch("torch.manual_seed", side_effect=ImportError("torch not found"))
def test_set_seeds_pytorch_not_installed(mock_manual_seed, mock_print):
    """Test that the function gracefully handles when Pytorch is not installed"""
    set_seeds(seed=123, pytorch_=True)



@patch("builtins.print")
@patch("tensorflow.random.set_seed", side_effect=ImportError("tensorflow not found"))
def test_set_seeds_tensorflow_not_installed(mock_tf_seed, mock_print):
    """Test that the function gracefully handles when TensorFlow is not installed"""
    set_seeds(seed=123, tensorflow_=True)



@patch("builtins.print")
@patch("numpy.random.seed", side_effect=ImportError("numpy not found"))
def test_set_seeds_numpy_not_installed(mock_numpy_seed, mock_print):
    """Test that the function gracefully handles when Numpy is not installed"""
    set_seeds(seed=123, numpy_=True)


