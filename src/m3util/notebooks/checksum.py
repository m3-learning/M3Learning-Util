import json
from m3util.util.hashing import select_hash_algorithm


def calculate_notebook_checksum(notebook_path, algorithm="sha256"):
    """
    Calculate the checksum of the code cells in a Jupyter notebook (.ipynb) file using the specified hashing algorithm.

    This function reads the contents of a Jupyter notebook file, extracts the code cells (ignoring metadata, execution counts,
    and other non-code content), and computes its checksum using one of the supported hashing algorithms: 'sha256', 'md5', or 'sha1'.

    Args:
        notebook_path (str): The path to the Jupyter notebook file (.ipynb) to be hashed.
        algorithm (str): The hashing algorithm to use. Supported options are 'sha256' (default), 'md5', and 'sha1'.

    Returns:
        str: The computed checksum as a hexadecimal string.

    Raises:
        ValueError: If an unsupported hashing algorithm is specified.

    Example:
        checksum = calculate_notebook_checksum("/path/to/notebook.ipynb", algorithm="sha256")
        print(checksum)
    """

    hash_function = select_hash_algorithm(algorithm)

    # Load the notebook as a JSON object
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Initialize an empty string to hold all code from code cells
    code_cells_content = ""

    # Loop over all cells in the notebook
    for cell in notebook["cells"]:
        # Check if the cell type is 'code'
        if cell["cell_type"] == "code":
            # Join the source lines of the code cell (if it's not empty)
            code_cells_content += "".join(cell.get("source", [])) + "\n"

    # Encode the concatenated code cells content as bytes and update the hash function
    hash_function.update(code_cells_content.encode("utf-8"))

    # Return the checksum as a hexadecimal string
    return hash_function.hexdigest()
