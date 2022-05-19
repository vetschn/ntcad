"""_summary_
"""

from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_nearest_neighbors(matrix: np.ndarray) -> Axes:
    """TODO: decide if this is actually useful."""
    num_atoms = matrix.shape[0]
    neighbors = matrix[:, 3:].astype(int) - 1
    neighbor_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for i, neighbor_indices in enumerate(neighbors):
        for neighbor_index in neighbor_indices:
            neighbor_matrix[i, neighbor_index] = 1
    return plt.spy(neighbor_matrix)
