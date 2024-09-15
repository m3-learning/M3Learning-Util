import hashlib
import json

def calculate_notebook_checksum(notebook_path, algorithm="sha256"):
    """
    Calculate the checksum of a Jupyter notebook (.ipynb) file using the specified hashing algorithm.
    
    This function reads the contents of a Jupyter notebook file and computes its checksum using
    one of the supported hashing algorithms: 'sha256', 'md5', or 'sha1'.
    
    Args:
        notebook_path (str): The path to the Jupyter notebook file (.ipynb) to be hashed.
        algorithm (str): The hashing algorithm to use. Supported options are 'sha256' (default), 'md5', and 'sha1'.
    
    Returns:
        str: The computed checksum as a hexadecimal string.
    
    Raises:
        ValueError: If an unsupported hashing algorithm is specified.
    """
    
    # Open the notebook file in read mode with UTF-8 encoding and read its contents as a string
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = f.read()

    # Select the hashing algorithm based on the user input
    if algorithm == "sha256":
        hash_function = hashlib.sha256()
    elif algorithm == "md5":
        hash_function = hashlib.md5()
    elif algorithm == "sha1":
        hash_function = hashlib.sha1()
    else:
        # Raise an error if the algorithm is not supported
        raise ValueError("Unsupported hashing algorithm. Use 'sha256', 'md5', or 'sha1'.")

    # Encode the notebook content as bytes and update the hash function with it
    hash_function.update(notebook_content.encode('utf-8'))
    
    # Return the checksum as a hexadecimal string
    return hash_function.hexdigest()
