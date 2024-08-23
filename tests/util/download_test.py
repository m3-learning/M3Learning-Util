import os
import pytest
from unittest import mock
from m3learning_util.util.IO import download, make_folder  # Adjust the import according to your file structure

@mock.patch('m3learning_util.util.IO.os.path.exists')
@mock.patch('m3learning_util.util.IO.os.remove')
@mock.patch('m3learning_util.util.IO.wget.download')
def test_download_file_does_not_exist(mock_wget_download, mock_os_remove, mock_os_path_exists):
    # Simulate that the file does not exist
    mock_os_path_exists.return_value = False

    url = "https://filesampleshub.com/download/document/txt/sample1.txt"
    destination = "/path/to/file.txt"

    # Run the download function
    download(url, destination)

    # Assert that wget.download was called
    mock_wget_download.assert_called_once_with(url, destination)
    # Assert that os.remove was not called (since the file did not exist)
    mock_os_remove.assert_not_called()

@mock.patch('m3learning_util.util.IO.os.path.exists')
@mock.patch('m3learning_util.util.IO.os.remove')
@mock.patch('m3learning_util.util.IO.wget.download')
def test_download_file_exists_no_force(mock_wget_download, mock_os_remove, mock_os_path_exists):
    # Simulate that the file exists
    mock_os_path_exists.return_value = True

    url = "https://filesampleshub.com/download/document/txt/sample1.txt"
    destination = "/path/to/file.txt"

    # Run the download function with force=False (default)
    download(url, destination)

    # Assert that wget.download was not called (since the file exists and force is False)
    mock_wget_download.assert_not_called()
    # Assert that os.remove was not called
    mock_os_remove.assert_not_called()

@mock.patch('m3learning_util.util.IO.os.path.exists')
@mock.patch('m3learning_util.util.IO.os.remove')
@mock.patch('m3learning_util.util.IO.wget.download')
def test_download_file_exists_with_force(mock_wget_download, mock_os_remove, mock_os_path_exists):
    # Simulate that the file exists
    mock_os_path_exists.return_value = True

    url = "https://filesampleshub.com/download/document/txt/sample1.txt"
    destination = "/path/to/file.txt"

    # Run the download function with force=True
    download(url, destination, force=True)

    # Assert that os.remove was called
    mock_os_remove.assert_called_once_with(destination)
    # Assert that wget.download was called
    mock_wget_download.assert_called_once_with(url, destination)
    
def test_make_folder_creates_directory(tmp_path):
    # Create a temporary directory path
    folder_name = tmp_path / "test_folder"
    
    # Ensure the folder doesn't exist yet
    assert not folder_name.exists()
    
    # Call the make_folder function
    result = make_folder(str(folder_name))
    
    # Check if the folder was created
    assert folder_name.exists()
    assert folder_name.is_dir()
    
    # Check if the returned path is correct
    assert result == str(folder_name)

def test_make_folder_existing_directory(tmp_path):
    # Create a directory first
    folder_name = tmp_path / "existing_folder"
    folder_name.mkdir()
    
    # Ensure the folder exists
    assert folder_name.exists()
    assert folder_name.is_dir()
    
    # Call the make_folder function
    result = make_folder(str(folder_name))
    
    # Ensure the folder still exists and no error was raised
    assert folder_name.exists()
    assert folder_name.is_dir()
    
    # Check if the returned path is correct
    assert result == str(folder_name)

@mock.patch('m3learning_util.util.IO.os.makedirs')
def test_make_folder_called_with_correct_args(mock_makedirs):
    folder_name = "mock_folder"
    
    # Call the make_folder function
    result = make_folder(folder_name)
    
    # Check if os.makedirs was called with the correct arguments
    mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)
    
    # Check if the returned path is correct
    assert result == folder_name
