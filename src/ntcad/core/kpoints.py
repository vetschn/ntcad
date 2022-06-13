"""
Utility functions for the creation of k-point grids.

"""

import numpy as np


def monkhorst_pack(size: np.ndarray) -> np.ndarray:
    """Constructs a uniform sampling of k-space of given size.

    Parameters
    ----------
    size
        Size of the Monkhorst-Pack grid.

    Returns
    -------
        An array containing all Monkhorst-Pack grid points.

    """
    kpoints = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpoints + 0.5) / size - 0.5


def kpoint_path(points: np.ndarray, num: int = 50) -> np.ndarray:
    """Generates a k-point path along the given symmetry points.

    Parameters
    ----------
    points
        Symmetry points along the path (``N_p`` x 3), where ``N_p`` is the
        number of symmetry points.
    num, optional
        The number of k-points along each section, by default 50.

    Returns
    -------
    kpoints
        All k-points along the given symmetry points (``N_s``*``num`` x 3),
        where ``N_s`` is the number of sections between symmetry points.
    """
    N_s = len(points) - 1
    sections = np.zeros((N_s, num, 3))
    for i in range(N_s):
        sections[i] = np.linspace(points[i], points[i + 1], num)
    kpoints = sections.reshape((N_s * num, 3))
    return kpoints
