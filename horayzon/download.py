# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
from tqdm import tqdm
import requests
import horayzon


# -----------------------------------------------------------------------------

def file(file_url, path_local):
    """Download file from web.

    Download file from web and show progress with bar.

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
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 10
        # download seems to be faster with larger block size...
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB",
                            unit_scale=True)
        with open(path_local + os.path.split(file_url)[-1], "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ValueError("Inconsistency in file size")
    except Exception:
        print("Download failed (probably because URL does not exist)")


# -----------------------------------------------------------------------------

def files(files_url, path_local):
    """Download multiple files from web.

    Download multiple files from web and show progress with bar.

    Parameters
    ----------
    files_url : str
        URL of file to download
    path_local: str
        Local path for downloaded file"""

    a = 20
