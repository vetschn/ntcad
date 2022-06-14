"""
Visualization routines for Wannier90 outputs.

"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


def plot_operator(
    O_R: np.ndarray,
    axis: int = 2,
    indices: int = 0,
    mod: Callable = np.abs,
    norm: Normalize = LogNorm(),
    **kwargs: dict,
) -> None:
    """Plots an operator.

    Parameters
    ----------
    O_R
        The operator to plot (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``), where ``N_i`` correspond to the number of
        Wigner-Seitz cells along the lattice vectors ``A_i``. The indices
        are chosen such that (0, 0, 0) actually gets you the center
        Wigner-Seitz cell.
    axis
        Which Wigner-Seitz index axis to fix, by default 2, i. e. the z
        axis.
    indices
        At which index to fix the selected Wigner-Seitz index axis, by
        default 0.
    mod
        A modifier to be applied to the matrix elements before plotting,
        by default np.abs.
    norm
        A normalizing function to be applied during plotting.

    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")

    # Midpoint of the Wigner-Seitz cell indices.
    midpoint = np.floor_divide(np.subtract(O_R.shape[:3], 1), 2)
    # Shift the operator to the center.
    O_ = np.zeros_like(O_R)
    for R in np.ndindex(O_R.shape[:3]):
        O_[(*R,)] = O_R[(*(R - midpoint),)]

    # Take operator blocks along one axis and apply modifier.
    O_ = np.take(mod(O_), indices=indices, axis=axis)
    # Concatenate the operator Wigner-Seitz cell blocks together.
    O = np.concatenate([np.concatenate(block, axis=1) for block in O_], axis=0)

    # Plotting.
    ax = kwargs.pop("ax", None)
    if ax is None:
        __, ax = plt.subplots()

    # The extent keyword here is used to set the ticks "correctly" and
    # to compensate for the fact that ax.matshow plots the pixels *on
    # top of* the index / coordinate.
    ax.matshow(O, norm=norm, extent=(0, O_.shape[0], O_.shape[1], 0), **kwargs)

    ax.set_xticks(np.arange(O_.shape[0]))
    ax.set_yticks(np.arange(O_.shape[0]))
    midpoint = np.floor_divide(np.subtract(O_.shape[:2], 1), 2)
    ax.set_xticklabels(np.arange(-midpoint[0], midpoint[0] + 1))
    ax.set_yticklabels(np.arange(-midpoint[1], midpoint[1] + 1))
    ax.grid(which="both")
