# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import horayzon


# -----------------------------------------------------------------------------

def file(file_url, path_local):
    """Download single file from web.

    Download single file from web and show progress with bar.

    Parameters
    ----------
    file_url : str
        URL of file to download
    path_local: str
        Local path for downloaded file"""

    # Check arguments
    if not os.path.isdir(path_local):
        raise ValueError("Local path does not exist")

    # Try to download file
    try:
        response = requests.get(file_url, stream=True)
    except requests.exceptions.SSLError:
        print("SSL certificate verification failed - continue download "
              + "(yes/no)?")
        cont = ""
        flag = False
        while cont not in ("yes", "no"):
            if flag:
                print("Please enter 'yes' or 'no'")
            cont = input()
            flag = True
        if cont == "yes":
            response = requests.get(file_url, stream=True, verify=False)
        else:
            return
    if response.ok:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 10
        # download seems to be faster with larger block size...
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB",
                            unit_scale=True)
        with open(path_local + os.path.split(file_url)[-1], "wb") as infile:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                infile.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ValueError("Inconsistency in file size")
    else:
        raise ValueError("URL does not exist")


# -----------------------------------------------------------------------------

def files(files_url, path_local, mode="parallel", block_size=500, file_num=10):
    """Download multiple files from web.

    Download multiple files from web in parallel.

    Parameters
    ----------
    files_url : list
        List with URLs of files to download
    path_local: str
        Local path for downloaded files
    mode: str
        Download mode (parallel or serial)
    block_size: int
        Block size for files that are processed in one step
    file_num: int (optional)
        Number of files that are downloaded in parallel"""

    # Check arguments
    if not os.path.isdir(path_local):
        raise ValueError("Local path does not exist")
    if mode not in ("serial", "parallel"):
        raise ValueError("Download mode must be either 'serial' or 'parallel'")

    # Try to download files
    downloaded = []
    files_count = 0
    if mode == "parallel":
        for i in range(0, len(files_url), block_size):
            executor = ThreadPoolExecutor(file_num)
            downloaded.extend(executor.map(get_file,
                                           files_url[i:i + block_size],
                                           repeat(path_local)))
            executor.shutdown()
            files_count += len(files_url[i:i + block_size])
            print("Files attempted to download: " + str(files_count) + "/"
                  + str(len(files_url)))
    else:
        for i in range(0, len(files_url), block_size):
            downloaded.extend([get_file(i, path_local)
                               for i in files_url[i:i + block_size]])
            files_count += len(files_url[i:i + block_size])
            print("Files attempted to download: " + str(files_count) + "/"
                  + str(len(files_url)))

    return downloaded


def get_file(file_url, path_local):
    try:
        response = requests.get(file_url, stream=True)
    except Exception:
        return False
    if response.ok:
        block_size = 1024 * 10
        # download seems to be faster with larger block size...
        with open(path_local + os.path.split(file_url)[-1], "wb") as infile:
            for data in response.iter_content(block_size):
                infile.write(data)
        return True
    else:
        return False
