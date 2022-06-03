# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
from scipy import interpolate


###############################################################################

def geoid_undulation(lon_ip, lat_ip, geoid="EGM96", path=None):
    """Compute geoid undulation.

    Compute the geoid undulation for the EGM96 or GEOID12A geoid by bilinear
    interpolation from gridded data.

    Parameters
    ----------
    lon_ip : ndarray of double
        Array (1-dimensional) with geographic longitude [degree]
    lat_ip : ndarray of double
        Array (1-dimensional) with geographic longitude [degree]
    geoid: str
        Geoid model (EGM96 or GEOID12A)
    path: str
        Path to file 'WW15MGH.GRD' (EGM96) or file 'g2012a00.asc' (GEOID12A)
        containing the undulation data for the geoid model

    Returns
    -------
    data_ip : ndarray of double
        Array (2-dimensional) with geoid undulation [m]

    Notes
    -----
    Source of geoid undulation data:
    - EGM96: https://earth-info.nga.mil/php/
             download.php?file=egm-96interpolation
    - GEOID12A: https://www.ngs.noaa.gov/PC_PROD/GEOID12A/Format_ascii/
                g2012aa0.asc.gz"""

    # Spatial coverage of data
    spat_cov = {"EGM96":    (-180.0, 180.0, -90.0, 90.0),
                "GEOID12A": (-180.0, -126.0, 49.0, 72.0)}

    # Check arguments
    if geoid not in ("EGM96", "GEOID12A"):
        raise NotImplementedError("geoid " + geoid + " is not supported")
    if path is None:
        raise ValueError("no path specified for geoid model data")
    if (lon_ip.min() < spat_cov[geoid][0]
            or lon_ip.max() > spat_cov[geoid][1]
            or lat_ip.min() < spat_cov[geoid][2]
            or lat_ip.max() > spat_cov[geoid][3]):
        raise ValueError("selected domain exceeds spatial coverage")
    if not np.all(np.diff(lon_ip) > 0.0):
        raise ValueError("longitude values are not monotonically increasing")
    if sum((np.all(np.diff(lat_ip) > 0.0),
            np.all(np.diff(lat_ip) < 0.0))) != 1:
        raise ValueError("longitude values are not monotonically increasing or"
                         "decreasing")

    # Check availability of geoid undulation data
    file = {"EGM96": "WW15MGH.GRD", "GEOID12A": "g2012a00.asc"}
    if not os.path.exists(path + file[geoid]):
        raise ImportError("Geoid undulation data not found in specified "
                          "directory")

    # Ensure that latitude values are monotonically increasing
    lat_dec = False
    if lat_ip[1] < lat_ip[0]:
        lat_dec = True
        lat_ip = lat_ip[::-1]

    # Compute geoid undulation
    data_ip = np.empty((len(lat_ip), len(lon_ip)), dtype=np.float64)
    # -------------------------------------------------------------------------
    if geoid == "EGM96":

        # Load data
        data = np.fromfile(path + file[geoid], sep=" ", dtype=np.float32)[6:]
        data = data.reshape(int(180 / 0.25) + 1, int(360 / 0.25) + 1)

        # Construct grid
        lon = np.linspace(0.0, 360.0, data.shape[1], dtype=np.float32)
        lat = np.linspace(90.0, -90, data.shape[0], dtype=np.float32)

        # Rearrange data (longitude: -180.0 -> +180.0)
        lon_ra = np.append(lon[720:] - 360.0, lon[1:721])
        data_ra = np.hstack((data[:, 720:], data[:, 1:721]))

        # Perform interpolation
        f_ip = interpolate.RectBivariateSpline(lat[::-1], lon_ra,
                                               np.flipud(data_ra), kx=1, ky=1)
        data_ip[:] = f_ip(lat_ip, lon_ip)  # [m]
    # -------------------------------------------------------------------------
    else:  # GEOID12A

        # Load data
        data = np.fromstring("".join(open(path + file[geoid], "r")
                                     .read().splitlines()), dtype=np.float32,
                             sep=" ")[7:]
        data = data.reshape(1381, 3721)

        # Construct grid
        lon = np.linspace(-188.0,
                          -188.0 + 0.1666666666667E-01 * (data.shape[1] - 1),
                          data.shape[1], dtype=np.float32)
        lat = np.linspace(49.0,
                          49.0 + 0.1666666666667E-01 * (data.shape[0] - 1),
                          data.shape[0], dtype=np.float32)

        # Perform interpolation
        f_ip = interpolate.RectBivariateSpline(lat, lon, data, kx=1, ky=1)
        data_ip[:] = f_ip(lat_ip, lon_ip)  # [m]
    # -------------------------------------------------------------------------

    if lat_dec:
        data_ip = np.flipud(data_ip)

    return data_ip
