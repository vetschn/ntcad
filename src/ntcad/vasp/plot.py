"""
Visualization routines for VASP outputs.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def bands(vasprun: dict, path: np.ndarray = None, **kwargs) -> Axes:
    """Plots the band structure obtained from a VASP run.

    Parameters
    ----------
    vasprun : dict
        Dictionary representing the vasprun.xml file, which contains the
        eigenvalues for each k-point. This dictionary can be obtained
        from the function ntcad.vasp.io.read_vasprun_xml.
    path : ndarray
        List of k-points to be plotted. If None, the path is obtained
        from the vasprun.xml file.
    **kwargs
        Keyword arguments to pass to matplotlib.

    See Also
    --------
    ntcad.vasp.io.read_vasprun_xml : Read vasprun.xml file.

    Returns
    -------
    ax : Axes
        The matplotlib axes object.

    """
    ax = kwargs.pop("ax", None)
    if ax is None:
        __, ax = plt.subplots()

    eigenvalues_section = vasprun["modeling"]["calculation"]["eigenvalues"]
    eigenvalues = eigenvalues_section["array"]["set"]["set"]["set"]

    num_kpoints = len(eigenvalues)
    num_bands = len(eigenvalues[0]["r"])
    bands = np.zeros((num_kpoints, num_bands))

    # Split eigenvalues from occupancy info and reshape.
    for i in range(num_kpoints):
        e, __ = zip(*[eigenvalue.split() for eigenvalue in eigenvalues[i]["r"]])
        e = list(map(float, e))
        bands[i] = e

    kpoints = range(num_kpoints)
    if path is not None:
        if not num_kpoints % (len(path) - 1) == 0:
            raise ValueError("Given path cannot be cast onto number of k-points.")

        # Make plotted distances proportional to reciprocal-space
        # distances between given symmetry points.
        num = int(num_kpoints / (len(path) - 1))
        kpoints = []
        d_total = 0.0

        for a, b in zip(path, path[1:]):
            d = np.linalg.norm(a - b)
            kpoints.append(np.linspace(d_total, d_total + d, num, endpoint=False))
            ax.axvline(x=d_total, c="k", lw=0.5)
            d_total += d
        ax.axvline(x=d_total, c="k", lw=0.5)

        kpoints = np.array(kpoints).flatten()

    ax.plot(kpoints, bands, **kwargs)

    return ax
