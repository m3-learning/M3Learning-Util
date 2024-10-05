import pytest
import subprocess
from unittest.mock import patch
from m3util.globus.globus import (
    check_globus_endpoint,
    check_globus_file_access,
    GlobusAccessError,
)

# Sample data to use for mocking
mock_endpoint_active = '{"is_paused": false, "is_connected": true}'
mock_endpoint_inactive = '{"is_paused": true, "is_connected": false}'
mock_error_response = "Error: some error occurred"


def test_check_globus_endpoint_active():
    """Test for an active endpoint."""
    # Mock the subprocess.run call to simulate the Globus CLI response for an active endpoint
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["globus", "endpoint", "show"],
            returncode=0,
            stdout=mock_endpoint_active,
            stderr="",
        )
        result = check_globus_endpoint("12345")
        assert result == "Endpoint 12345 is active"


def test_check_globus_endpoint_inactive():
    """Test for an inactive endpoint."""
    # Mock the subprocess.run call to simulate the Globus CLI response for an inactive endpoint
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["globus", "endpoint", "show"],
            returncode=0,
            stdout=mock_endpoint_inactive,
            stderr="",
        )
        result = check_globus_endpoint("12345")
        assert result == "Endpoint 12345 is not active"


def test_check_globus_endpoint_error():
    """Test for an endpoint showing an error."""
    # Mock the subprocess.run call to simulate an error response from the Globus CLI
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["globus", "endpoint", "show"],
            returncode=1,
            stdout="",
            stderr=mock_error_response,
        )
        result = check_globus_endpoint("12345")
        assert result == f"Error: {mock_error_response}"


def test_check_globus_file_access_success():
    """Test for successful access to a file or directory."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess to simulate successful file listing
        mock_run.return_value = subprocess.CompletedProcess(
            args=["globus", "ls"],
            returncode=0,
            stdout="file1.txt\nfile2.txt\n",
            stderr="",
        )
        result = check_globus_file_access("12345", "/path/to/file", verbose=True)
        assert mock_run.called




def test_check_globus_file_access_success_verbose():
    """Test for successful access to a file or directory with verbose output."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess to simulate successful file listing
        mock_run.return_value = subprocess.CompletedProcess(
            args=["globus", "ls"],
            returncode=0,
            stdout="file1.txt\nfile2.txt\n",
            stderr="",
        )
        with patch("builtins.print") as mock_print:
            result = check_globus_file_access("12345", "/path/to/file", verbose=True)
            assert mock_run.called
            mock_print.assert_called_with(
                "Access to '/path/to/file' confirmed.\nOutput:\nfile1.txt\nfile2.txt\n"
            )


def test_check_globus_endpoint_exception():
    """Test for an exception during the endpoint check."""
    with patch("subprocess.run") as mock_run:
        # Simulate an exception being raised during the subprocess call
        mock_run.side_effect = Exception("Subprocess failed")
        result = check_globus_endpoint("12345")
        assert result == "Exception occurred: Subprocess failed"
