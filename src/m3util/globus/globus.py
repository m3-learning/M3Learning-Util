import subprocess
import json
import os


def check_globus_endpoint(endpoint_id):
    """
    Check the status of a Globus endpoint by its ID.

    This function uses the Globus CLI to check whether a given endpoint is active
    or not. It parses the output in JSON format to determine the endpoint's
    connection status.

    Args:
        endpoint_id (str): The ID of the Globus endpoint to check.

    Returns:
        str: A message indicating whether the endpoint is active or not,
        or an error message if the check fails.

    Raises:
        Exception: If the subprocess or JSON parsing fails.
    """
    try:
        # Call the Globus CLI to get the endpoint status in JSON format
        result = subprocess.run(
            ["globus", "endpoint", "show", endpoint_id, "--format", "json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the subprocess call was successful
        if result.returncode == 0:
            # Parse the JSON output
            endpoint_info = json.loads(result.stdout)

            # Check if the endpoint is paused or disconnected
            if endpoint_info.get("is_paused", True) or not endpoint_info.get(
                "is_connected", False
            ):
                return f"Endpoint {endpoint_id} is not active"
            else:
                return f"Endpoint {endpoint_id} is active"
        else:
            # Return error message from stderr if subprocess fails
            return f"Error: {result.stderr}"
    except Exception as e:
        # Handle exceptions related to subprocess or JSON parsing
        return f"Exception occurred: {str(e)}"


class GlobusAccessError(Exception):
    """Custom exception for Globus file access errors."""

    pass


def check_globus_file_access(endpoint_id, file_path, verbose=False):
    """
    Check whether a file or directory is accessible via a Globus endpoint.

    This function uses the Globus CLI to list the content of a file or directory
    at a given path associated with a specific Globus endpoint. If the path is
    accessible, it prints the output or raises an error if access fails.

    Args:
        endpoint_id (str): The ID of the Globus endpoint.
        file_path (str): The path of the file or directory to check.
        verbose (bool): Whether to print detailed information about the access check.
                        Defaults to False.

    Returns:
        None

    Raises:
        GlobusAccessError: If the file access fails, an error message is raised.
    """
    try:
        # Convert the local file path to an absolute path
        file_path = os.path.abspath(file_path)

        # Run the Globus CLI command to list the file or directory
        result = subprocess.run(
            ["globus", "ls", f"{endpoint_id}:{file_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the subprocess call was successful
        if result.returncode == 0:
            # If verbose is enabled, print the access confirmation and output
            if verbose:
                print(f"Access to '{file_path}' confirmed.\nOutput:\n{result.stdout}")
        else:
            # Raise a custom GlobusAccessError if the access check fails
            raise GlobusAccessError(
                f"Error accessing '{file_path}': {result.stderr.strip()}"
            )
    except Exception as e:
        # Handle exceptions related to subprocess
        raise GlobusAccessError(
            f"Exception occurred while accessing '{file_path}': {str(e)}"
        )
