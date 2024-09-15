import subprocess
import json
import os 

def check_globus_endpoint(endpoint_id):
    try:
        # Call the Globus CLI to get the endpoint status
        result = subprocess.run(
            ['globus', 'endpoint', 'show', endpoint_id, '--format', 'json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # If the call was successful, parse the JSON output
        if result.returncode == 0:
            endpoint_info = json.loads(result.stdout)
            if endpoint_info.get('is_paused', True) or not endpoint_info.get('is_connected', False):
                return f"Endpoint {endpoint_id} is not active"
            else:
                return f"Endpoint {endpoint_id} is active"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

class GlobusAccessError(Exception):
    """Custom exception for Globus file access errors."""
    pass

def check_globus_file_access(endpoint_id, file_path, verbose=False):
    try:
        # Convert local path to full path
        file_path = os.path.abspath(file_path)

        # Run the Globus CLI command to list the file or directory
        result = subprocess.run(
            ['globus', 'ls', f'{endpoint_id}:{file_path}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if verbose: 
            print(f"Access to '{file_path}' confirmed.\nOutput:\n{result.stdout}")
    except Exception as e:
        raise GlobusAccessError(f"Error accessing '{file_path}': {result.stderr.strip()}")
