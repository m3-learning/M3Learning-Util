import hashlib


# TODO -- add a search for similar checksum and reference
def calculate_h5file_checksum(file_path, algorithm="sha256", chunk_size=1048576):
    """
    Calculate the checksum of an HDF5 (.h5) file using the specified hashing algorithm.

    This function reads the binary content of the HDF5 file and computes its checksum using
    one of the supported hashing algorithms: 'sha256', 'md5', or 'sha1'.

    Args:
        file_path (str): The path to the HDF5 file (.h5) to be hashed.
        algorithm (str): The hashing algorithm to use. Supported options are 'sha256' (default), 'md5', and 'sha1'.

    Returns:
        str: The computed checksum as a hexadecimal string.

    Raises:
        ValueError: If an unsupported hashing algorithm is specified.
    """

    hash_function = select_hash_algorithm(algorithm)

    # Open the file in binary mode and compute its checksum
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):  # Read the file in chunks (8192 bytes)
            hash_function.update(chunk)

    # Return the checksum as a hexadecimal string
    return hash_function.hexdigest()


def select_hash_algorithm(algorithm):
    """
    Select the hashing algorithm based on the user input.

    Args:
        algorithm (str): The name of the hashing algorithm to use. Supported options are 'sha256', 'md5', and 'sha1'.

    Returns:
        hashlib._hashlib.HASH: An instance of the selected hashing algorithm.

    Raises:
        ValueError: If an unsupported hashing algorithm is specified.
    """
    # Check the algorithm and return the corresponding hash function
    if algorithm == "sha256":
        hash_function = hashlib.sha256()
    elif algorithm == "md5":
        hash_function = hashlib.md5()
    elif algorithm == "sha1":
        hash_function = hashlib.sha1()
    else:
        # Raise an error if the algorithm is not supported
        raise ValueError(
            "Unsupported hashing algorithm. Use 'sha256', 'md5', or 'sha1'."
        )
    return hash_function
