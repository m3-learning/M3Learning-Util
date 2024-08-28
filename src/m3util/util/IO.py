import wget
import os
import sys
import time
import urllib
import zipfile
import shutil
import os.path
from os.path import exists
import csv
from tqdm import tqdm
import requests


def download(url: str, destination: str, force: bool = False) -> None:
    """
    Downloads a file from a URL to a destination.
    If the file already exists, it can optionally overwrite it.

    Parameters:
        url (str): The URL of the file to download.
        destination (str): The local path where the file should be saved.
        force (bool, optional): Whether to force download and overwrite. Default is False.

    Returns:
        None
    """
    # Check if the file already exists
    if os.path.exists(destination):
        if force:
            # If force is True, overwrite the existing file
            print(f"Overwriting existing file {destination}...")
            os.remove(destination)
        else:
            # If force is False, skip the download
            print(f"The file {destination} already exists. Skipping download.")
            return

    # Download the file from the given URL to the destination
    print(f"Downloading file from {url} to {destination}...")
    wget.download(url, destination)
    print("\nDownload complete.")


def make_folder(folder, **kwargs):
    """Utility to make folders

    Args:
        folder (string): name of folder

    Returns:
        string: path to folder
    """
    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return folder

def download_files_from_txt(url_file, 
                            download_path):
    """Download files from URLs listed in a text file.
    
    Args:
    url_file (str): Path to the text file containing URLs, each on a new line.
    download_path (str): Directory to save the downloaded files. The directory must exist.
    
    """
    # create folder if not yet
    make_folder(download_path)
    abs_path = os.path.abspath(download_path)
    
    # set delay
    delay = 1
    
    # Open the text file containing URLs
    with open(url_file, 'r') as file:
        urls = file.readlines()

    # Iterate over each URL
    for url in tqdm(urls):
        url = url.strip()  # Remove any extraneous whitespace or newline characters
        if url:  # Ensure the URL is not empty
            while True:
                try:
                    # Make HTTP GET request to the URL
                    response = requests.get(url, stream=True)
                    response.raise_for_status()  # Check if the request was successful

                    # Extract filename from URL if possible, or default to a name with its index
                    filename = url.split('/')[-1]
                    # skip download if file exists
                    if os.path.exists(f'{abs_path}/{filename}'):
                        print(f"File already exists: {filename}")
                        break
                    file_path = os.path.join(abs_path, filename)

                    # Save the content to a file in the specified download path
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded: {filename}")
                    break

                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429:  # Too Many Requests
                        print("Rate limit reached, waiting to retry...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        if delay > 1024:
                            print(f"Failed to download {url}: time exceeds limit")
                            break

                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {url}: {str(e)}")
                    break  # exit the loop if a different HTTP error occurred


def reporthook(count, block_size, total_size):
    """
    A function that displays the status and speed of the download

    Args:
        count (int): The number of blocks downloaded.
        block_size (int): The size of each block in bytes.
        total_size (int): The total size of the file in bytes.
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration + 0.0001))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


# TODO: This could be refactored with the above download function
def download_file(url, filename):
    """A function that downloads the data file from a URL

    Args:
        url (string): url where the file to download is located
        filename (string): location where to save the file
    """
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename, reporthook)


def compress_folder(base_name, format, root_dir=None):
    """Function that zips a folder can save zip and tar

    Args:
        base_name (string): base name of the zip file
        format (string): sets the format of the zip file. Can either be zip or tar
        root_dir (string, optional): sets the root directory to save the file. Defaults to None.
    """
    shutil.make_archive(base_name, format, root_dir)


def unzip(filename, path):
    """Function that unzips the files


    Args:
        filename (string): base name of the zip file
        path (string): path where the zip file will be saved
    """
    zip_ref = zipfile.ZipFile("./" + filename, "r")
    zip_ref.extractall(path)
    zip_ref.close()


def get_size(start_path="."):
    """A function that computes the size of a folder


    Args:
        start_path (str, optional): Path to compute the size of. Defaults to '.'.

    Returns:
        float: Size of the folder
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def download_and_unzip(filename, url, save_path, force=False):
    """Function that computes the size of a folder

    Args:
        filename (str): filename to save the zip file
        url (str): url where the file is located
        save_path (str): place where the data is saved
        download_data (bool, optional): sets if to download the data. Defaults to True.
    """
    make_folder(save_path)

    path = save_path + "/" + filename
    # if np.int(get_size(save_path) / 1e9) < 1:
    if exists(path) and not force:
        print("Using files already downloaded")
    else:
        print("downloading data")
        download_file(url, path)

    if ".zip" in filename:
        if os.path.isfile(path):
            print(f"extracting {path}")
            unzip(path, save_path)


def append_to_csv(file_path, data, headers):
    """
    Appends a row of data to a CSV file.

    If the file doesn't exist, it creates a new file and writes the header row.

    Args:
        file_path (str): The path to the CSV file.
        data (list): A list of values representing a row of data to be appended.
        headers (list): A list of header values for the CSV file.
    """
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)  # Write header row if the file is newly created
        writer.writerow(data)
