# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np


###############################################################################

def pad_geometry_buffer(buffer):
    """Padding of geometry buffer.

    Pads geometric buffer to make it conformal with 16-byte SSE load
    instructions.

    Parameters
    ----------
    buffer : ndarray
        Array (1-dimensional) with geometry buffer [arbitrary]

    Returns
    -------
    buffer : ndarray
        Array (1-dimensional) with padded geometry buffer [arbitrary]

    Notes
    -----
    This function ensures that vertex buffer size is divisible by 16 and hence
    conformal with 16-byte SSE load instructions (see Embree documentation;
    section 7.45 rtcSetSharedGeometryBuffer)."""

    # Check arguments
    if not isinstance(buffer, np.ndarray):
        raise ValueError("argument 'buffer' has invalid type")
    if buffer.ndim != 1:
        raise ValueError("argument 'buffer' must be one-dimensional")

    add_elem = 16
    if not (buffer.nbytes % 16) == 0:
        add_elem += ((16 - (buffer.nbytes % 16)) // buffer[0].nbytes)
    buffer = np.append(buffer, np.zeros(add_elem, dtype=buffer.dtype))

    return buffer
