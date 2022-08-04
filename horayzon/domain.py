# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from geographiclib.geodesic import Geodesic


# -----------------------------------------------------------------------------

def planar_grid(domain, dist_search=50.0):
    """Compute digital elevation model domain (planar grid).

    Computes total required domain of digital elevation model for planar grid.

    Parameters
    ----------
    domain : dict
        Dictionary with domain boundaries (x_min, x_max, y_min, y_max) [metre]
    dist_search : float
        Search distance for horizon [kilometre]

    Returns
    -------
    domain_outer : dict
        Dictionary with outer domain boundaries (x_min, x_max, y_min, y_max)
        [metre]"""

    # Check arguments
    if ((domain["x_min"] >= domain["x_max"])
            or (domain["y_min"] >= domain["y_max"])):
        raise ValueError("Invalid domain specification")

    # Compute outer domain boundaries
    domain_outer = {"x_min": domain["x_min"] - (dist_search * 1000.0),
                    "x_max": domain["x_max"] + (dist_search * 1000.0),
                    "y_min": domain["y_min"] - (dist_search * 1000.0),
                    "y_max": domain["y_max"] + (dist_search * 1000.0)}

    return domain_outer


# -----------------------------------------------------------------------------

def curved_grid(domain, dist_search=50.0, ellps="sphere"):
    """Compute digital elevation model domain (curved grid).

    Computes total required domain of digital elevation model for curved grid.

    Parameters
    ----------
    domain : dict
        Dictionary with domain boundaries (lon_min, lon_max, lat_min, lat_max)
        [degree]
    dist_search : float
        Search distance for horizon [kilometre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    domain_outer : dict
        Dictionary with outer domain boundaries (lon_min, lon_max,
        lat_min, lat_max) [degree]

    Notes
    -----
    Source:
    - Geoid parameters a and f: PROJ
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    # Check arguments
    if ellps not in ("sphere", "GRS80", "WGS84"):
        raise NotImplementedError("ellipsoid " + ellps + " is not supported")
    if ((domain["lon_min"] >= domain["lon_max"])
            or (domain["lat_min"] >= domain["lat_max"])):
        raise ValueError("Invalid domain specification")

    # Initialise geodesic
    if ellps == "sphere":
        a = 6370997.0  # earth radius [m]
        f = 0.0  # flattening [-]
    elif ellps == "GRS80":
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        f = (1.0 / 298.257222101)  # flattening [-]
    else:
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        f = (1.0 / 298.257223563)  # flattening [-]
    b = a * (1.0 - f)  # polar radius (semi-minor axis) [m]
    e_2 = 1.0 - (b ** 2 / a ** 2)  # squared num. eccentricity [-]
    geod = Geodesic(a, f)

    # Compute outer domain boundaries
    lat_abs_max = np.maximum(np.abs(domain["lat_min"]),
                             np.abs(domain["lat_max"]))
    rad_sph = a / np.sqrt(1.0 - e_2 * np.sin(np.deg2rad(lat_abs_max)) ** 2) \
        * np.cos(np.deg2rad(lat_abs_max))  # sphere radius [m]
    lon_add = 360.0 / (2.0 * np.pi * rad_sph) * (dist_search * 1000.0)  # [deg]
    domain_outer = {"lon_min": domain["lon_min"] - lon_add,
                    "lon_max": domain["lon_max"] + lon_add,
                    "lat_min": geod.Direct(domain["lat_min"], 0.0, 180.0,
                                           dist_search * 1000.0)["lat2"],
                    "lat_max": geod.Direct(domain["lat_max"], 0.0, 0.0,
                                           dist_search * 1000.0)["lat2"]}

    # Check if total domain is within valid range (lon: -/+180.0, lat: -/+90.0)
    if ((domain_outer["lon_min"] < -180.0)
            or (domain_outer["lon_max"] > 180.0)
            or (domain_outer["lat_min"] < -90.0)
            or (domain_outer["lat_max"] > 90.0)):
        raise ValueError("total domain exceeds valid range")

    return domain_outer
