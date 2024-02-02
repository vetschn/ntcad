"""
Visualization routines for CP2K data.

"""

from typing import Callable

import numpy as np
import pyvista as pv


def basis_function(
    psi: Callable,
    n: int,
    plotter=None,
    square_modulus=False,
    grid_size=50,
    zoom=1.0,
    isosurfaces=100,
):
    """Plots a basis function in 3D.

    Parameters
    ----------
    psi : Callable
        The basis function to plot. This function should take three
        arguments `x`, `y`, and `z` and return the value of the basis
        function at that point.
    n : int
        The principal quantum number of the basis function.
    plotter : pyvista.Plotter, optional
        The plotter to use. If not provided, a new plotter will be
        created and returned.
    square_modulus : bool, optional
        If True, the square modulus of the basis function will be
        plotted. Otherwise, the basis function itself will be plotted.
    grid_size : int, optional
        The number of grid points in each dimension. The default is 50.
    zoom : float, optional
        The zoom factor. The default is 1.0.
    isosurfaces : int, optional
        The number of isosurfaces to plot. By default, 100 isosurfaces
        will be plotted.

    Returns
    -------
    pyvista.Plotter
        The plotter used to create the plot. You need to call
        `plotter.show()` to actually display the plot.

    """
    origin = (1.5 * n**2 + 10.0) / zoom
    sp = (origin * 2) / (grid_size - 1)

    grid = pv.UniformGrid(
        dimensions=(grid_size, grid_size, grid_size),
        spacing=(sp, sp, sp),
        origin=(-origin, -origin, -origin),
    )

    wfc = psi(grid.x, grid.y, grid.z).reshape(grid.dimensions)

    if plotter is None:
        plotter = pv.Plotter(notebook=False)

    if square_modulus:
        grid["square_modulus"] = np.abs(wfc.ravel()) ** 2
        plotter.add_mesh(
            grid.contour(isosurfaces=isosurfaces, scalars="square_modulus"),
            cmap="plasma",
            smooth_shading=True,
            opacity=[0, 1],
        )
        return plotter

    grid["wave_function"] = wfc.ravel()
    plotter.add_mesh(
        grid.contour(isosurfaces=isosurfaces, scalars="wave_function"),
        cmap="bwr",
        smooth_shading=True,
        opacity=[1, 0, 1],
    )
    return plotter
