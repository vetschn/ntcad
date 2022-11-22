"""
**k-points**
============
Functions for generating Monkhorst-Pack k-point grids and k-point paths.

"""

import numpy as np


def monkhorst_pack(size: np.ndarray) -> np.ndarray:
    """Generates a Monkhorst-Pack [1] k-point grid.

    Parameters
    ----------
    size
        The size of the k-point grid. This has to be an array of three
        integers, giving the number of k-points along each lattice
        vector.

    Returns
    -------
    np.ndarray
        The Monkhorst-Pack k-point grid points in fractional
        coordinates.

    References
    ----------
    .. [1] Monkhorst, H. J., & Pack, J. D. (1976). *Special points for
           Brillouin-zone integrations*, Phys. Rev. B, 13(12), 5188-5192

    """
    if len(size) != 3:
        raise ValueError("size must be a 3D vector")
    kpoints = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpoints + 0.5) / size - 0.5


def kpoint_path(points: np.ndarray, num: int = 50) -> np.ndarray:
    """Generates a k-point path along the given symmetry points.

    Parameters
    ----------
    points
        The symmetry points along the path. This has to be an array of
        shape (N, 3), N being the number of symmetry points. The
        symmetry points are given in fractional coordinates.
    num
        The number of k-points along each section of the path, by
        default 50.

    Returns
    -------
    np.ndarray
        The k-points along the path, given in fractional coordinates.

    """
    N_s = len(points) - 1
    sections = np.zeros((N_s, num, 3))
    for i in range(N_s):
        sections[i] = np.linspace(points[i], points[i + 1], num)
    kpoints = sections.reshape((N_s * num, 3))
    return kpoints
