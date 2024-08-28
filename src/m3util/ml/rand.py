import random
import os
import numpy as np

try:
    import torch
except:
    print('torch not found')

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


# def set_seeds(seed=42):
#     """
#     Sets the random seeds for reproducibility.

#     Args:
#         seed (int): The random seed value.
#     """
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


def set_seeds(seed = 42, pytorch_ = True, numpy_ = True, tensorflow_ = True):
    """Function that sets the random seed

    Args:
        seed (int, optional): value for the seed. Defaults to 42.
        pytorch_ (bool, optional): chooses if you set the pytorch seed. Defaults to True.
        numpy_ (bool, optional): chooses if you set the numpy seed. Defaults to True.
        tensorflow_ (bool, optional): chooses if you set the tensorflow seed. Defaults to True.
    """    
    
    try:
        if pytorch_:
            import torch
            # torch.set_default_dtype(torch.float64)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            np.random.seed(seed)  # Numpy module.
            random.seed(seed)  # Python random module.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print(f'Pytorch seed was set to {seed}')
    except: 
        pass
    
    try:
        np.random.seed(42)
        print(f'Numpy seed was set to {seed}')
    except: 
        pass
    
    try: 
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f'tensorflow seed was set to {seed}')
    except: 
        pass
