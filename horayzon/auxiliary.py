# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
from tqdm import tqdm
import requests
import horayzon


# -----------------------------------------------------------------------------

def get_path_aux_data():
    """Get path for auxiliary data.

        Get path for auxiliary data. Read from text file in 'HORAYZON' main
        directory if already defined, otherwise define by user.

    Returns
    -------
    path_aux_data: str
        Path of auxiliary data"""

    # Create text file with path to auxiliary data
    file_name = "path_aux_data.txt"
    path_horayzon = os.path.join(os.path.split(
        os.path.dirname(horayzon.__file__))[0], "")
    if not os.path.isfile(path_horayzon + "/" + file_name):
        valid_path = False
        print("Provide path for auxiliary data:")
        while not valid_path:
            path_aux_data = os.path.join(input(), "")
            if os.path.isdir(path_aux_data):
                valid_path = True
            else:
                print("Provided path is invalid - try again:")
        file = open(path_horayzon + "/" + file_name, "w")
        file.write(path_aux_data)
        file.close()
    else:
        file = open(path_horayzon + "/" + file_name, "r")
        path_aux_data = file.read()
        file.close()

    return path_aux_data


# -----------------------------------------------------------------------------

def download_file(file_url, file_local):
    """Download file from web.

    Download file from web and show progress with bar.

    Parameters
    ----------
    file_url : str
        URL of file to download
    file_local: str
        Local path and name for downloaded file"""

    # Check arguments
    if not os.path.isdir(os.path.split(file_local)[0]):
        raise ValueError("Local path does not exist")

    # Try to download file
    try:
        response = requests.get(file_url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 10
        # download seems to be faster with larger block size...
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB",
                            unit_scale=True)
        with open(file_local, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ValueError("Inconsistency in file size")
    except Exception:
        print("Download failed (probably because URL does not exist)")


# -----------------------------------------------------------------------------

def rearrange_pad_buffer(x, y, z):
    """Rearrange elevation model data and pad geometry buffer.

    Rearrange digital elevation model data and pad geometric buffer to make it
    conformal with 16-byte SSE load instructions.

    Parameters
    ----------
    x : ndarray of float
        Array (two-dimensional) with x-coordinates of digital elevation model
        data [arbitrary]
    y : ndarray of float
        Array (two-dimensional) with y-coordinates of digital elevation model
        data [arbitrary]
    z : ndarray of float
        Array (two-dimensional) with z-coordinates of digital elevation model
        data [arbitrary]

    Returns
    -------
    buffer : ndarray of float
        Array (one-dimensional) with padded geometry buffer [arbitrary]

    Notes
    -----
    This function ensures that vertex buffer size is divisible by 16 and hence
    conformal with 16-byte SSE load instructions (see Embree documentation;
    section 7.45 rtcSetSharedGeometryBuffer)."""

    # Check arguments
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or not isinstance(z, np.ndarray)):
        raise TypeError("One or more input arguments are of invalid type")
    if ((x.dtype != np.float32) or (y.dtype != np.float32)
            or (z.dtype != np.float32)):
        raise TypeError("Not all input arguments are 32-bit floats")
    if (any([i != 2 for i in [x.ndim, y.ndim, z.ndim]])
            or not x.shape == y.shape == z.shape):
        raise ValueError("Dimensions of input arguments are "
                         "erroneous/inconsistent")

    # Rearrange digital elevation model data
    buffer = np.hstack((x.reshape(x.size, 1), y.reshape(x.size, 1),
                        z.reshape(x.size, 1))).ravel()

    # Padding of geometry buffer
    add_elem = 16
    if not (buffer.nbytes % 16) == 0:
        add_elem += ((16 - (buffer.nbytes % 16)) // buffer[0].nbytes)
    buffer = np.append(buffer, np.zeros(add_elem, dtype=buffer.dtype))

    return buffer
