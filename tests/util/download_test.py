import os
import pytest
from unittest import mock
from m3learning_util.util.download import download  # Adjust the import according to your file structure

@mock.patch('m3learning_util.util.download.os.path.exists')
@mock.patch('m3learning_util.util.download.os.remove')
@mock.patch('m3learning_util.util.download.wget.download')
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

@mock.patch('m3learning_util.util.download.os.path.exists')
@mock.patch('m3learning_util.util.download.os.remove')
@mock.patch('m3learning_util.util.download.wget.download')
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

@mock.patch('m3learning_util.util.download.os.path.exists')
@mock.patch('m3learning_util.util.download.os.remove')
@mock.patch('m3learning_util.util.download.wget.download')
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
