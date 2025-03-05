"""
Visualization routines for CP2K data.

"""

import warnings
from typing import Callable

import numpy as np
import pyvista as pv


def basis_function(
    psi: Callable,
    plotter=None,
    square_modulus=False,
    grid_size=50,
    spacing=0.1,
    isosurfaces=100,
    origin=None,
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
    if origin is None:
        origin = [0] * 3

    origin = [component - grid_size * spacing / 2 for component in origin]

    grid = pv.ImageData(
        dimensions=(grid_size, grid_size, grid_size),
        spacing=(spacing, spacing, spacing),
        origin=origin,
    )

    wfc = psi(grid.x, grid.y, grid.z).reshape(grid.dimensions)

    if plotter is None:
        plotter = pv.Plotter(notebook=False)

    grid["wave_function"] = wfc.ravel().real + wfc.ravel().imag
    grid["square_modulus"] = np.abs(wfc.ravel()) ** 2
    if square_modulus:
        integral = np.sum(grid["square_modulus"]) * spacing**3
        if not np.isclose(integral, 1.0, atol=1e-3):
            warnings.warn(
                f"Integral of the square modulus is not 1.0 (it is {integral:.3f})."
            )

        plotter.add_mesh(
            grid.contour(isosurfaces=isosurfaces, scalars="square_modulus"),
            cmap="plasma",
            clim=[0, np.max(grid["square_modulus"])],
            smooth_shading=True,
            opacity=[0, 1],
        )
        return plotter

    plotter.add_mesh(
        grid.contour(isosurfaces=isosurfaces, scalars="wave_function"),
        clim=[-np.max(grid["wave_function"]), np.max(grid["wave_function"])],
        cmap="bwr",
        smooth_shading=True,
        opacity=[1, 0, 1],
    )
    return plotter
