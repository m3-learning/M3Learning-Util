import random
import torch
import os
import numpy as np


def rand_tensor(min=0, max=1, size=(1)):
    """
    Generates a random tensor within a specified range.

    Args:
        min (float): The minimum value of the tensor.
        max (float): The maximum value of the tensor.
        size (tuple): The size of the random tensor to generate.

    Returns:
        torch.Tensor: The random tensor.
    """
    out = (max - min) * torch.rand(size) + min
    return out


def set_seeds(seed=42):
    """
    Sets the random seeds for reproducibility.

    Args:
        seed (int): The random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
