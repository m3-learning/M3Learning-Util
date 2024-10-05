from unittest import mock
from m3util.util.IO import (
    download,
    make_folder,
    reporthook,
    download_file,
    compress_folder,
    unzip,
    get_size,
    download_and_unzip,
    append_to_csv,
    download_files_from_txt,
)  # Adjust the import according to your file structure
import pytest
import os
import requests
from unittest.mock import MagicMock, patch, mock_open


@mock.patch("m3util.util.IO.os.path.exists")
@mock.patch("m3util.util.IO.os.remove")
@mock.patch("m3util.util.IO.wget.download")
def test_download_file_does_not_exist(
    mock_wget_download, mock_os_remove, mock_os_path_exists
):
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


@mock.patch("m3util.util.IO.os.path.exists")
@mock.patch("m3util.util.IO.os.remove")
@mock.patch("m3util.util.IO.wget.download")
def test_download_file_exists_no_force(
    mock_wget_download, mock_os_remove, mock_os_path_exists
):
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


@mock.patch("m3util.util.IO.os.path.exists")
@mock.patch("m3util.util.IO.os.remove")
@mock.patch("m3util.util.IO.wget.download")
def test_download_file_exists_with_force(
    mock_wget_download, mock_os_remove, mock_os_path_exists
):
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


@mock.patch("m3util.util.IO.os.makedirs")
def test_make_folder_called_with_correct_args(mock_makedirs):
    folder_name = "mock_folder"

    # Call the make_folder function
    result = make_folder(folder_name)

    # Check if os.makedirs was called with the correct arguments
    mock_makedirs.assert_called_once_with(folder_name, exist_ok=True)

    # Check if the returned path is correct
    assert result == folder_name


def test_reporthook():
    # Initial call to simulate the start of download
    reporthook(0, 1024, 10240)

    # Initial call to simulate the middle of download
    reporthook(5, 1024, 10240)


@mock.patch("m3util.util.IO.urllib.request.urlretrieve")
@mock.patch("m3util.util.IO.os.path.isfile")
def test_download_file_when_file_exists(mock_isfile, mock_urlretrieve):
    # Simulate that the file already exists
    mock_isfile.return_value = True

    url = "https://filesampleshub.com/download/document/txt/sample1.txt"
    filename = "/path/to/file.txt"

    # Call the download_file function
    download_file(url, filename)

    # Assert that os.path.isfile was called with the correct filename
    mock_isfile.assert_called_once_with(filename)

    # Assert that urllib.request.urlretrieve was not called, since the file exists
    mock_urlretrieve.assert_not_called()


@mock.patch("m3util.util.IO.urllib.request.urlretrieve")
@mock.patch("m3util.util.IO.os.path.isfile")
def test_download_file_when_file_does_not_exist(mock_isfile, mock_urlretrieve):
    # Simulate that the file does not exist
    mock_isfile.return_value = False

    url = "https://filesampleshub.com/download/document/txt/sample1.txt"
    filename = "/path/to/file.txt"

    # Call the download_file function
    download_file(url, filename)

    # Assert that os.path.isfile was called with the correct filename
    mock_isfile.assert_called_once_with(filename)

    # Assert that urllib.request.urlretrieve was called with the correct arguments
    mock_urlretrieve.assert_called_once_with(url, filename, mock.ANY)


@mock.patch("m3util.util.IO.shutil.make_archive")
def test_compress_folder_zip(mock_make_archive):
    # Test parameters
    base_name = "test_archive"
    format = "zip"
    root_dir = "/path/to/folder"

    # Call the compress_folder function
    compress_folder(base_name, format, root_dir)

    # Assert that shutil.make_archive was called with the correct arguments
    mock_make_archive.assert_called_once_with(base_name, format, root_dir)


@mock.patch("m3util.util.IO.shutil.make_archive")
def test_compress_folder_tar(mock_make_archive):
    # Test parameters
    base_name = "test_archive"
    format = "tar"
    root_dir = "/path/to/folder"

    # Call the compress_folder function
    compress_folder(base_name, format, root_dir)

    # Assert that shutil.make_archive was called with the correct arguments
    mock_make_archive.assert_called_once_with(base_name, format, root_dir)


@mock.patch("m3util.util.IO.shutil.make_archive")
def test_compress_folder_default_root_dir(mock_make_archive):
    # Test parameters
    base_name = "test_archive"
    format = "zip"

    # Call the compress_folder function with the default root_dir
    compress_folder(base_name, format)

    # Assert that shutil.make_archive was called with the correct arguments
    mock_make_archive.assert_called_once_with(base_name, format, None)


@mock.patch("m3util.util.IO.zipfile.ZipFile")
def test_unzip(mock_zipfile):
    # Mock instance of ZipFile
    mock_zip_ref = mock_zipfile.return_value

    # Test parameters
    filename = "test_archive.zip"
    path = "/path/to/extract"

    # Call the unzip function
    unzip(filename, path)

    # Assert that zipfile.ZipFile was called with the correct arguments
    mock_zipfile.assert_called_once_with("./" + filename, "r")

    # Assert that extractall was called with the correct path
    mock_zip_ref.extractall.assert_called_once_with(path)

    # Assert that close was called once
    mock_zip_ref.close.assert_called_once()


@mock.patch("m3util.util.IO.os.path.getsize")
@mock.patch("m3util.util.IO.os.walk")
def test_get_size(mock_os_walk, mock_getsize):
    # Define the mock return values for os.walk and os.path.getsize

    # Simulate a directory structure with files
    mock_os_walk.return_value = [
        ("/path/to/dir", ("subdir",), ("file1.txt", "file2.txt")),
        ("/path/to/dir/subdir", (), ("file3.txt",)),
    ]

    # Simulate the file sizes returned by os.path.getsize
    mock_getsize.side_effect = [
        100,
        200,
        300,
    ]  # Sizes of file1.txt, file2.txt, file3.txt respectively

    # Call the get_size function
    total_size = get_size("/path/to/dir")

    # Check that os.walk was called with the correct directory
    mock_os_walk.assert_called_once_with("/path/to/dir")

    # Check that os.path.getsize was called with the correct file paths
    mock_getsize.assert_any_call("/path/to/dir/file1.txt")
    mock_getsize.assert_any_call("/path/to/dir/file2.txt")
    mock_getsize.assert_any_call("/path/to/dir/subdir/file3.txt")

    # Check that the total size is correct
    assert total_size == 600  # 100 + 200 + 300


@mock.patch("m3util.util.IO.unzip")
@mock.patch("m3util.util.IO.os.path.isfile")
@mock.patch("m3util.util.IO.download_file")
@mock.patch("m3util.util.IO.exists")
@mock.patch("m3util.util.IO.make_folder")
def test_download_and_unzip_existing_file(
    mock_make_folder, mock_exists, mock_download_file, mock_isfile, mock_unzip
):
    # Test parameters
    filename = "data.zip"
    url = "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip"
    save_path = "/path/to/save"

    # Simulate that the file already exists and force=False
    mock_exists.return_value = True
    mock_isfile.return_value = True

    # Call the download_and_unzip function
    download_and_unzip(filename, url, save_path, force=False)

    # Ensure make_folder is called
    mock_make_folder.assert_called_once_with(save_path)

    # Ensure download_file is not called since the file exists and force=False
    mock_download_file.assert_not_called()

    # Ensure unzip is called since the file exists
    mock_unzip.assert_called_once_with(save_path + "/" + filename, save_path)


@mock.patch("m3util.util.IO.unzip")
@mock.patch("m3util.util.IO.os.path.isfile")
@mock.patch("m3util.util.IO.download_file")
@mock.patch("m3util.util.IO.exists")
@mock.patch("m3util.util.IO.make_folder")
def test_download_and_unzip_force_download(
    mock_make_folder, mock_exists, mock_download_file, mock_isfile, mock_unzip
):
    # Test parameters
    filename = "data.zip"
    url = "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip"
    save_path = "/path/to/save"

    # Simulate that the file exists but force=True
    mock_exists.return_value = True
    mock_isfile.return_value = True

    # Call the download_and_unzip function with force=True
    download_and_unzip(filename, url, save_path, force=True)

    # Ensure make_folder is called
    mock_make_folder.assert_called_once_with(save_path)

    # Ensure download_file is called despite the file existing
    mock_download_file.assert_called_once_with(url, save_path + "/" + filename)

    # Ensure unzip is called
    mock_unzip.assert_called_once_with(save_path + "/" + filename, save_path)


@mock.patch("m3util.util.IO.unzip")
@mock.patch("m3util.util.IO.os.path.isfile")
@mock.patch("m3util.util.IO.download_file")
@mock.patch("m3util.util.IO.exists")
@mock.patch("m3util.util.IO.make_folder")
def test_download_and_unzip_file_does_not_exist(
    mock_make_folder, mock_exists, mock_download_file, mock_isfile, mock_unzip
):
    # Test parameters
    filename = "data.zip"
    url = "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip"
    save_path = "/path/to/save"

    # Simulate that the file does not exist
    mock_exists.return_value = False
    mock_isfile.return_value = True

    # Call the download_and_unzip function
    download_and_unzip(filename, url, save_path, force=False)

    # Ensure make_folder is called
    mock_make_folder.assert_called_once_with(save_path)

    # Ensure download_file is called since the file does not exist
    mock_download_file.assert_called_once_with(url, save_path + "/" + filename)

    # Ensure unzip is called
    mock_unzip.assert_called_once_with(save_path + "/" + filename, save_path)


@mock.patch("m3util.util.IO.os.path.isfile")
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("m3util.util.IO.csv.writer")
def test_append_to_csv_file_does_not_exist(mock_csv_writer, mock_open, mock_isfile):
    # Simulate that the file does not exist
    mock_isfile.return_value = False

    # Test parameters
    file_path = "test.csv"
    data = ["value1", "value2", "value3"]
    headers = ["header1", "header2", "header3"]

    # Call the append_to_csv function
    append_to_csv(file_path, data, headers)

    # Ensure that os.path.isfile is called with the correct file path
    mock_isfile.assert_called_once_with(file_path)

    # Ensure the file is opened in append mode
    mock_open.assert_called_once_with(file_path, "a", newline="")

    # Get the mock writer object and ensure the correct rows are written
    mock_writer = mock_csv_writer.return_value
    mock_writer.writerow.assert_any_call(
        headers
    )  # Header should be written since the file does not exist
    mock_writer.writerow.assert_any_call(data)  # Data row should be written


@mock.patch("m3util.util.IO.os.path.isfile")
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("m3util.util.IO.csv.writer")
def test_append_to_csv_file_exists(mock_csv_writer, mock_open, mock_isfile):
    # Simulate that the file exists
    mock_isfile.return_value = True

    # Test parameters
    file_path = "test.csv"
    data = ["value1", "value2", "value3"]
    headers = ["header1", "header2", "header3"]

    # Call the append_to_csv function
    append_to_csv(file_path, data, headers)

    # Ensure that os.path.isfile is called with the correct file path
    mock_isfile.assert_called_once_with(file_path)

    # Ensure the file is opened in append mode
    mock_open.assert_called_once_with(file_path, "a", newline="")

    # Get the mock writer object and ensure only the data row is written
    mock_writer = mock_csv_writer.return_value
    mock_writer.writerow.assert_called_once_with(
        data
    )  # Only data row should be written


@pytest.fixture
def sample_url_file(tmp_path):
    # Create a temporary text file with some URLs
    url_file = tmp_path / "urls.txt"
    with url_file.open("w") as f:
        f.write("http://example.com/file1.txt\n")
        f.write("http://example.com/file2.txt\n")
    return str(url_file)


def test_download_files_success(tmp_path, sample_url_file):
    download_path = tmp_path / "downloads"
    # Mock response for requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"Test content"]
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        download_files_from_txt(sample_url_file, str(download_path))

        # Check that directory was created
        assert download_path.exists()

        # Check that files are created
        file1 = download_path / "file1.txt"
        file2 = download_path / "file2.txt"
        assert file1.exists()
        assert file2.exists()

        # Check the content of the files
        with file1.open("rb") as f:
            content = f.read()
            assert content == b"Test content"
        with file2.open("rb") as f:
            content = f.read()
            assert content == b"Test content"


def test_download_files_existing_files(tmp_path, sample_url_file):
    download_path = tmp_path / "downloads"
    os.makedirs(download_path)

    # Create one of the files in advance
    file1 = download_path / "file1.txt"
    with file1.open("wb") as f:
        f.write(b"Existing content")

    # Mock response for requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"New content"]
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        download_files_from_txt(sample_url_file, str(download_path))

        # Check that file1.txt was not overwritten
        with file1.open("rb") as f:
            content = f.read()
            assert content == b"Existing content"

        # Check that file2.txt was created
        file2 = download_path / "file2.txt"
        assert file2.exists()
        with file2.open("rb") as f:
            content = f.read()
            assert content == b"New content"


def test_download_files_empty_urls(tmp_path):
    download_path = tmp_path / "downloads"

    # Create a URL file with empty lines
    url_file = tmp_path / "urls.txt"
    with url_file.open("w") as f:
        f.write("\n")
        f.write("http://example.com/file1.txt\n")
        f.write("\n")
        f.write("http://example.com/file2.txt\n")
        f.write("\n")

    # Mock response for requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"Test content"]
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        download_files_from_txt(str(url_file), str(download_path))
        # Check that files were created
        file1 = download_path / "file1.txt"
        file2 = download_path / "file2.txt"
        assert file1.exists()
        assert file2.exists()


def test_download_files_create_directory(tmp_path, sample_url_file):
    download_path = tmp_path / "downloads"
    # Do not create the directory
    assert not download_path.exists()

    # Mock response for requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"Test content"]
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        download_files_from_txt(sample_url_file, str(download_path))

        # Check that directory was created
        assert download_path.exists()
        # Check that files are created
        file1 = download_path / "file1.txt"
        file2 = download_path / "file2.txt"
        assert file1.exists()
        assert file2.exists()

@patch("builtins.open", new_callable=mock_open)
@patch("os.path.exists")
@patch("os.path.abspath")
@patch("requests.get")
@patch("time.sleep")
def test_file_already_exists(
    mock_sleep, mock_requests_get, mock_abspath, mock_exists, mock_open_func
):
    """Test that the function skips downloading if the file already exists."""
    mock_exists.return_value = True  # Simulate that the file already exists
    mock_abspath.return_value = "/abs/path"  # Mock absolute path

    # Mock URL file contents
    with patch(
        "builtins.open", mock_open(read_data="http://example.com/file1.txt\n")
    ) as mock_file:
        download_files_from_txt("urls.txt", "downloads")



@patch("builtins.open", new_callable=mock_open)
@patch("os.path.exists")
@patch("os.path.abspath")
@patch("requests.get")
@patch("time.sleep")
def test_http_error_rate_limit(
    mock_sleep, mock_requests_get, mock_abspath, mock_exists, mock_open_func
):
    """Test that the function handles 429 Too Many Requests error with exponential backoff."""
    mock_exists.return_value = False  # Simulate that the file doesn't exist
    mock_abspath.return_value = "/abs/path"  # Mock absolute path

    # Simulate the 429 Too Many Requests HTTP error
    response_mock = MagicMock()
    response_mock.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response_mock
    )
    response_mock.status_code = 429
    mock_requests_get.return_value = response_mock

    # Mock URL file contents
    with patch(
        "builtins.open", mock_open(read_data="http://example.com/file1.txt\n")
    ) as mock_file:
        download_files_from_txt("urls.txt", "downloads")



@patch("builtins.open", new_callable=mock_open)
@patch("os.path.exists")
@patch("os.path.abspath")
@patch("requests.get")
def test_request_exception(
    mock_requests_get, mock_abspath, mock_exists, mock_open_func
):
    """Test that the function handles general request exceptions."""
    mock_exists.return_value = False  # Simulate that the file doesn't exist
    mock_abspath.return_value = "/abs/path"  # Mock absolute path

    # Simulate a general request exception (e.g., connection error)
    mock_requests_get.side_effect = requests.exceptions.RequestException(
        "Connection error"
    )

    # Mock URL file contents
    with patch(
        "builtins.open", mock_open(read_data="http://example.com/file1.txt\n")
    ) as mock_file:
        download_files_from_txt("urls.txt", "downloads")

