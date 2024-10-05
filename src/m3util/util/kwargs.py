import inspect


def _filter_kwargs(obj, kwargs):
    """Filters out invalid keyword arguments for a given function or method.

    Args:
        obj (callable): The function or method to filter keyword arguments for.
        kwargs (dict): The keyword arguments to filter.

    Returns:
        dict: A dictionary containing only the valid keyword arguments for the given function or method.
    """
    # Get the valid parameters for the given function or method
    valid_args = inspect.signature(obj).parameters

    # Filter out invalid keyword arguments
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

    return filtered_kwargs
