# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from geographiclib.geodesic import Geodesic

# -> add functions for planar domain and single locations!


# -----------------------------------------------------------------------------

def dem_domain_loc(loc, width_in, dist_search=50.0, ellps="sphere"):
    """Compute Digital Elevation model (DEM) domain.

    Computes required domain of Digital Elevation model (DEM) from location
    and width of inner domain (geodetic coordinates).

    Parameters
    ----------
    loc : tuple
        Tuple with geodetic latitude/longitude of centre [degree]
    width_in : float
        Total x/y-width of inner domain for which horizon is computed
        [kilometre]
    dist_search : float
        Search distance for horizon [kilometre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)

    Returns
    -------
    dom : dict
        Dictionary with boundaries of inner and total domain [degree]

    Notes
    -----
    Source:
    - Geoid parameters a and f: PROJ
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    # Check arguments
    if ellps not in ("sphere", "GRS80", "WGS84"):
        raise NotImplementedError("ellipsoid " + ellps + " is not supported")

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

    # Inner domain
    width_h = width_in / 2.0 * 1000.0  # [m]
    rad_sph = a / np.sqrt(1 - e_2 * np.sin(np.deg2rad(loc[0])) ** 2) \
        * np.cos(np.deg2rad(loc[0]))  # sphere radius [m]
    lon_add = (360.0 / (np.pi * 2 * rad_sph)) * width_h  # [deg]
    dom_in = {"lon_min": loc[1] - lon_add,
              "lon_max": loc[1] + lon_add,
              "lat_min": geod.Direct(loc[0], loc[1], 180.0, width_h)["lat2"],
              "lat_max": geod.Direct(loc[0], loc[1], 0.0, width_h)["lat2"]}

    # Total domain
    add_out = dist_search * 1000.0  # [m]
    lat_abs_max = max(abs(dom_in["lat_min"]), abs(dom_in["lat_max"]))
    rad_sph = a / np.sqrt(1 - e_2 * np.sin(np.deg2rad(lat_abs_max)) ** 2) \
        * np.cos(np.deg2rad(lat_abs_max))  # sphere radius [m]
    lon_add = (360.0 / (np.pi * 2 * rad_sph)) * add_out  # [deg]
    dom_tot = {"lon_min": dom_in["lon_min"] - lon_add,
               "lon_max": dom_in["lon_max"] + lon_add,
               "lat_min": geod.Direct(loc[0], loc[1], 180.0,
                                      width_h + add_out)["lat2"],
               "lat_max": geod.Direct(loc[0], loc[1], 0.0,
                                      width_h + add_out)["lat2"]}

    # Check if total domain is within valid range (lon: -/+180.0, lat: -/+90.0)
    if ((dom_tot["lon_min"] < -180.0) or (dom_tot["lon_max"] > 180.0)
            or (dom_tot["lat_min"] < -90.0) or (dom_tot["lat_max"] > 90.0)):
        raise ValueError("total domain exceeds valid range")

    return {"in": dom_in, "tot": dom_tot}
