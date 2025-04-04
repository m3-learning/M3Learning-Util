import inspect

def filter_kwargs(obj, kwargs):
    if isinstance(obj, type):
        return filter_cls_params(obj, kwargs)
    if callable(obj):
        return _filter_kwargs(obj, kwargs)
    raise ValueError(f"Invalid object type: {type(obj)}")

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


def filter_cls_params(cls, params):
    """Filters params dict to match the signature of cls.__init__."""
    signature = inspect.signature(cls.__init__)
    return {
        k: v for k, v in params.items()
        if k in signature.parameters and k != 'self'
    }
