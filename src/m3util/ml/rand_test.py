import torch
import random
import numpy as np
import os
from m3util.ml.rand import rand_tensor, set_seeds


def test_rand_tensor_values():
    """Test that rand_tensor generates a tensor within the specified range."""
    min_val = 0
    max_val = 5
    size = (1000,)
    tensor = rand_tensor(min=min_val, max=max_val, size=size)

    assert isinstance(tensor, torch.Tensor), "The output should be a torch.Tensor"
    assert torch.all(
        tensor >= min_val
    ), "All elements should be greater than or equal to the minimum value"
    assert torch.all(
        tensor < max_val
    ), "All elements should be less than the maximum value"


def test_rand_tensor_size():
    """Test that rand_tensor generates a tensor of the correct size."""
    size = (10, 20)
    tensor = rand_tensor(size=size)

    assert tensor.shape == size, f"Expected tensor shape {size}, but got {tensor.shape}"


def test_set_seeds():
    """Test that set_seeds sets the random seeds for reproducibility."""
    set_seeds(42)

    # Check reproducibility for random
    random_vals = [random.random() for _ in range(5)]
    set_seeds(42)
    random_vals_reproduced = [random.random() for _ in range(5)]
    assert random_vals == random_vals_reproduced, "Random values are not reproducible"

    # Check reproducibility for numpy
    np_vals = np.random.rand(5)
    set_seeds(42)
    np_vals_reproduced = np.random.rand(5)
    assert np.array_equal(
        np_vals, np_vals_reproduced
    ), "NumPy random values are not reproducible"

    # Check reproducibility for torch
    torch_vals = torch.rand(5)
    set_seeds(42)
    torch_vals_reproduced = torch.rand(5)
    assert torch.equal(
        torch_vals, torch_vals_reproduced
    ), "Torch random values are not reproducible"


def test_environment_seed():
    """Test that the PYTHONHASHSEED environment variable is set correctly."""
    seed = 123
    set_seeds(seed)
    assert os.environ["PYTHONHASHSEED"] == str(
        seed
    ), f"PYTHONHASHSEED should be set to {seed}"
