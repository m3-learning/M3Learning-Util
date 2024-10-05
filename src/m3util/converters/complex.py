import torch
import numpy as np


def to_complex(data):
    """
    Converts the input data to a complex number if it is not already complex.

    Args:
        data (any): Input data which can be a PyTorch tensor or a NumPy array.

    Returns:
        any: The input data converted to a complex number if it was not already complex.

    Raises:
        TypeError: If the input data is not a PyTorch tensor or a NumPy array.
        IndexError: If the input data is empty or not an array-like structure.
    """
    # Check if the input data is empty or not an array-like structure
    if not hasattr(data, "__getitem__") or len(data) == 0:
        raise IndexError("Input data is empty or not an array-like structure.")

    try:
        # Extract the first element
        data = data[0]

        if isinstance(data, torch.Tensor):
            # Check if PyTorch tensor is complex
            if not torch.is_complex(data):
                data = data.to(torch.complex64)
            return data

        elif isinstance(data, np.ndarray):
            # Check if NumPy array is complex
            if not np.iscomplexobj(data):
                data = data.astype(np.complex64)
            return data

        else:
            # Raise an error if the data type is unsupported
            raise TypeError(
                f"Unsupported data type: {type(data)}. Expected torch.Tensor or np.ndarray."
            )

    except IndexError:
        # Raise an error if the input data is empty or not array-like
        raise IndexError("Input data is empty or not an array-like structure.")

    except Exception as e:
        # Raise a runtime error for any other exceptions
        raise RuntimeError(f"An error occurred while processing the data: {str(e)}")
