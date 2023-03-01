"""
This module contains functions for visualizing the results of the
calculations.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def nearest_neighbors(matrix: np.ndarray, **kwargs: dict) -> Axes:
    """Plots the nearest neighbors of the given matrix.

    Parameters
    ----------
    matrix : ndarray
        The matrix to plot the nearest neighbors of.
    **kwargs : dict
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    ax : Axes
        The axes of the plot.

    See Also
    --------
    matplotlib.pyplot.spy : The plot function used to plot the data.

    """
    ax = kwargs.pop("ax")
    if ax is None:
        fig, ax = plt.subplots()

    num_atoms = matrix.shape[0]
    neighbors = matrix[:, 3:].astype(int) - 1
    neighbor_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for i, neighbor_indices in enumerate(neighbors):
        for neighbor_index in neighbor_indices:
            neighbor_matrix[i, neighbor_index] = 1
    ax.spy(neighbor_matrix)
    return ax
