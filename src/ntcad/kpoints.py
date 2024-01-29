"""
Functions for generating Monkhorst-Pack k-point grids and k-point paths.

"""

import numpy as np
from numpy.typing import ArrayLike


def monkhorst_pack(size: ArrayLike) -> np.ndarray:
    """Generates a Monkhorst-Pack [1]_ k-point grid.

    Parameters
    ----------
    size : array_like
        The size of the k-point grid. This is an array of three
        integers, specifying the number of k-points along each lattice
        vector.

    Returns
    -------
    grid : ndarray
        A The Monkhorst-Pack k-point grid points in fractional
        coordinates.

    References
    ----------
    .. [1] Monkhorst, H. J., & Pack, J. D. (1976). *Special points for
       Brillouin-zone integrations*, Phys. Rev. B, 13(12), 5188-5192

    Examples
    --------
    Create a 2x2x2 Monkhorst-Pack k-point grid:

    >>> monkhorst_pack([2, 2, 2])
    array([[-0.25, -0.25, -0.25],
           [-0.25, -0.25,  0.25],
           [-0.25,  0.25, -0.25],
           [-0.25,  0.25,  0.25],
           [ 0.25, -0.25, -0.25],
           [ 0.25, -0.25,  0.25],
           [ 0.25,  0.25, -0.25],
           [ 0.25,  0.25,  0.25]])

    """
    if len(size) != 3:
        raise ValueError("size must be an array of length 3")
    kpoints = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpoints + 0.5) / size - 0.5


def kpoint_path(points: ArrayLike, num_points: int = 50) -> np.ndarray:
    """Generates a k-point path along the given symmetry points.

    Parameters
    ----------
    points : array_like
        The symmetry points along the k-point path. This is an array of
        shape (N, 3), N being the number of symmetry points. The
        symmetry points are given in fractional coordinates.
    num_points : int, optional
        The number of k-points along each section of the path, by
        default 50.

    Returns
    -------
    kpoints : ndarray
        The k-points along the path, given in fractional coordinates.

    Examples
    --------
    Create a k-point path between the :math:`\\Gamma` point and the
    :math:`X` point with 9 k-points along the path:

    >>> kpoint_path([[0, 0, 0], [0.5, 0.5, 0]], num_points=9)
    array([[0.    , 0.    , 0.    ],
           [0.0625, 0.0625, 0.    ],
           [0.125 , 0.125 , 0.    ],
           [0.1875, 0.1875, 0.    ],
           [0.25  , 0.25  , 0.    ],
           [0.3125, 0.3125, 0.    ],
           [0.375 , 0.375 , 0.    ],
           [0.4375, 0.4375, 0.    ],
           [0.5   , 0.5   , 0.    ]])

    """
    num_sections = len(points) - 1
    sections = np.zeros((num_sections, num_points, 3))
    for i in range(num_sections):
        sections[i] = np.linspace(points[i], points[i + 1], num_points)
    kpoints = sections.reshape((num_sections * num_points, 3))
    return kpoints
