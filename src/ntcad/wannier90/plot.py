"""
Visualization routines for Wannier90 outputs.

"""

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LogNorm, Normalize

from ntcad.structure import Structure
from ntcad.utils import center_index
from ntcad.wannier90.io import read_xsf


def operator(
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
        The operator to plot (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann``
        x ``num_wann``), where ``N_i`` correspond to the number of
        Wigner-Seitz cells along the lattice vectors ``A_i``. The
        indices are chosen such that (0, 0, 0) actually gets you the
        center Wigner-Seitz cell.
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
    center = center_index(O_R.shape[:3])
    # Shift the operator to the center.
    O_ = np.zeros_like(O_R)
    for R in np.ndindex(O_R.shape[:3]):
        O_[tuple(R)] = O_R[tuple(np.subtract(R, center))]

    # Take operator blocks along one axis and apply modifier.
    O_ = np.take(mod(O_), indices=indices, axis=axis)
    # Concatenate the operator Wigner-Seitz cell blocks together.
    O = np.concatenate([np.concatenate(block, axis=1) for block in O_], axis=0)

    # Plotting.
    ax = kwargs.pop("ax", None)
    fig = kwargs.pop("fig", None)
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    # The extent keyword here is used to set the ticks "correctly" and
    # to compensate for the fact that ax.matshow plots the pixels *on
    # top of the index / coordinate.
    qm = ax.matshow(O, norm=norm, extent=(0, O_.shape[0], O_.shape[1], 0), **kwargs)
    fig.colorbar(qm, ax=ax)

    ax.set_xticks(np.arange(O_.shape[0]))
    ax.set_yticks(np.arange(O_.shape[0]))
    center = center_index(O_.shape)
    ax.set_xticklabels(np.arange(-center[0], center[0] + 1))
    ax.set_yticklabels(np.arange(-center[1], center[1] + 1))
    ax.grid(which="both")


def xsf(xsf: Any, **kwargs) -> pv.Plotter:
    """Plots an XCrysDen file read from Wannier90 using PyVista.

    Parameters
    ----------
    xsf : Any
        The data to plot. Can be a filename, a dictionary or an
        ``ntcad.Structure``.
    plotter : pv.Plotter
        The plotter to use. If ``None``, a new plotter is created.
    datagrid_cell : bool
        Whether to plot the datagrid cell, by default ``False``.

    Returns
    -------
    pv.Plotter
        The plotter object. A new plotter is created if ``plotter`` is
        ``None``.

    """
    if pv is None:
        raise ImportError("PyVista is not installed.")

    if isinstance(xsf, str):
        structure = read_xsf(xsf)
    elif isinstance(xsf, Structure):
        structure = xsf

    # Read the volume data.
    field = structure.attr.get("3D_field", None)
    if field is None:
        raise ValueError("No volume data found.")
    shape = field["shape"]
    cell = field["cell"]
    origin = field["origin"]
    data = field["data"]

    # Create the UniformGrid.
    grid = pv.UniformGrid()
    grid.dimensions = shape
    grid.spacing = [1.0 / s for s in shape]
    grid.point_data["data"] = data

    # Construct a transformation matrix to map the rectilinear grid onto
    # the actual cell.
    transform = np.eye(4)
    transform[:3, :3] = cell.T
    transform[:3, 3] = origin

    grid = grid.transform(transform, inplace=False)

    # Plotting.
    pl = kwargs.pop("plotter", None)
    if pl is None:
        pl = pv.Plotter()

    # Plot the structure.
    structure.view(viewer="pyvista", plotter=pl)

    # Plot the volume data.
    pl.add_mesh(
        grid.contour(kwargs.pop("isosurfaces", 250)),
        cmap=kwargs.pop("cmap", "coolwarm"),
        clim=kwargs.pop("clim", [-1.0, 1.0]),
        smooth_shading=kwargs.pop("smooth_shading", True),
        opacity=kwargs.pop("opacity", [1.0, 0.0, 1.0]),
    )

    if kwargs.pop("datagrid_cell", False):
        pl.add_mesh(
            grid.extract_feature_edges(),
            style="wireframe",
            color="black",
            line_width=2,
            opacity=0.5,
        )

    return pl
