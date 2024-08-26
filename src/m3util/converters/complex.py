import torch
import numpy as np

def to_complex(data):
    """
    to_complex function to check if data is complex. If not complex, makes it a complex number.

    Args:
        data (any): input data

    Returns:
        any: array or tensor as a complex number
    """
    try:
        # Check if data is a list or array-like and extract the first element
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
            raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or np.ndarray.")

    except IndexError:
        raise IndexError("Input data is empty or not an array-like structure.")

    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the data: {str(e)}")
