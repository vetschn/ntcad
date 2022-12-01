"""
Structure
=========

This module implements the structure class, defining a configuration of
atoms in a unit cell together with some useful methods.

"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import linalg as npla

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    import ase
    import ase.visualize
except ImportError:
    ase = None


# All allowed atomic symbols including a `None` / "X" kind.
ATOMIC_SYMBOLS = (
    # --- 0 ------------------------------------------------------------
    "X "
    # --- 1 ------------------------------------------------------------
    "H He "
    # --- 2 ------------------------------------------------------------
    "Li Be B C N O F Ne "
    # --- 3 ------------------------------------------------------------
    "Na Mg Al Si P S Cl Ar "
    # --- 4 ------------------------------------------------------------
    "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    # --- 5 ------------------------------------------------------------
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    # --- 6 ------------------------------------------------------------
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    # --- 7 ------------------------------------------------------------
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og "
).split()

# The atomic number mapped to the corresponding atomic symbol.
ATOMIC_NUMBERS = {}
for i, symbol in enumerate(ATOMIC_SYMBOLS):
    ATOMIC_NUMBERS[symbol] = i

# Useful dtype to represent atomic sites.
_sites_dtype = np.dtype(
    [
        ("kind", np.unicode_, 2),
        ("position", np.float64, 3),
    ]
)

# The jmol colors of each atomic kind
# https://jmol.sourceforge.net/jscolors/
_jmol_colors = {
    "X": "#000000",
    "H": "#FFFFFF",
    "He": "#D9FFFF",
    "Li": "#CC80FF",
    "Be": "#C2FF00",
    "B": "#FFB5B5",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Ne": "#B3E3F5",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Ar": "#80D1E3",
    "K": "#8F40D4",
    "Ca": "#3DFF00",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Ga": "#C28F8F",
    "Ge": "#668F8F",
    "As": "#BD80E3",
    "Se": "#FFA100",
    "Br": "#A62929",
    "Kr": "#5CB8D1",
    "Rb": "#702EB0",
    "Sr": "#00FF00",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Tc": "#3B9E9E",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Pd": "#006985",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "In": "#A67573",
    "Sn": "#668080",
    "Sb": "#9E63B5",
    "Te": "#D47A00",
    "I": "#940094",
    "Xe": "#429EB0",
    "Cs": "#57178F",
    "Ba": "#00C900",
    "La": "#70D4FF",
    "Ce": "#FFFFC7",
    "Pr": "#D9FFC7",
    "Nd": "#C7FFC7",
    "Pm": "#A3FFC7",
    "Sm": "#8FFFC7",
    "Eu": "#61FFC7",
    "Gd": "#45FFC7",
    "Tb": "#30FFC7",
    "Dy": "#1FFFC7",
    "Ho": "#00FF9C",
    "Er": "#00E675",
    "Tm": "#00D452",
    "Yb": "#00BF38",
    "Lu": "#00AB24",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
    "Re": "#267DAB",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#D0D0E0",
    "Au": "#FFD123",
    "Hg": "#B8B8D0",
    "Tl": "#A6544D",
    "Pb": "#575961",
    "Bi": "#9E4FB5",
    "Po": "#AB5C00",
    "At": "#754F45",
    "Rn": "#428296",
    "Fr": "#420066",
    "Ra": "#007D00",
    "Ac": "#70ABFA",
    "Th": "#00BAFF",
    "Pa": "#00A1FF",
    "U": "#008FFF",
    "Np": "#0080FF",
    "Pu": "#006BFF",
    "Am": "#545CF2",
    "Cm": "#785CE3",
    "Bk": "#8A4FE3",
    "Cf": "#A136D4",
    "Es": "#B31FD4",
    "Fm": "#B31FBA",
    "Md": "#B30DA6",
    "No": "#BD0D87",
    "Lr": "#C70066",
    "Rf": "#CC0059",
    "Db": "#D1004F",
    "Sg": "#D90045",
    "Bh": "#E00038",
    "Hs": "#E6002E",
    "Mt": "#EB0026",
}


class Structure:
    """
    An arrangement of atoms in space.

    Attributes
    ----------
    kinds : numpy.ndarray
        The atomic kinds of the structure. Each kind is a 2-character
        string.
    positions : numpy.ndarray
        The atomic positions of the structure. Each position is a 3D
        vector.
    sites : numpy.ndarray
        The atomic sites and positions in one array. Each site is a
        2-tuple of the kind and position.
    cell : numpy.ndarray
        The cell vectors of the structure. Each cell vector is a 3D
        vector.
    attr : dict
        A dictionary of attributes of the structure. These are
        arbitrary key-value pairs that can be used to store
        additional information about the structure.

    """

    def __init__(
        self,
        kinds: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
        cartesian: bool = True,
        attr: dict = None,
    ) -> None:
        """
        Initializes a structure.

        Parameters
        ----------
        kinds
            The atomic kinds as a list of 2-character strings.
        positions
            The atomic positions as a list of 3D vectors.
        cell
            The cell vectors as a list of 3D vectors. If the cell vectors
            span an invalid cell, an exception is raised.
        cartesian
            Whether the positions are given in Cartesian coordinates. By
            default, the positions are assumed to be in Cartesian
            coordinates.
        attr
            A dictionary of attributes of the structure. These are
            arbitrary key-value pairs that can be used to store
            additional information about the structure.

        """
        self.kinds = np.array(kinds)
        for kind in self.kinds:
            if not isinstance(kind, str):
                raise ValueError("Invalid atomic kind")
            if kind not in ATOMIC_SYMBOLS:
                raise ValueError(f"Invalid atomic symbol: {kind}")

        self.cell = np.array(cell)
        if np.isclose(self.volume, 0.0):
            raise ValueError("Cell volume is zero")

        self.positions = np.array(positions)
        if not cartesian:
            self.positions = self.positions @ self.cell

        self.sites = np.array(list(zip(kinds, positions)), dtype=_sites_dtype)
        self.attr = attr

    def __repr__(self) -> str:
        return f"Structure({self.kinds}, {self.positions}, {self.cell})"

    @property
    def volume(self) -> float:
        """
        float: The volume of the structure's cell.
        """
        a1, a2, a3 = self.cell
        return np.dot(a1, np.cross(a2, a3))

    @property
    def reciprocal_cell(self) -> float:
        """
        float: The reciprocal cell vectors of the structure.
        """
        return 2 * np.pi * np.transpose(npla.inv(self.cell))

    def view(self, viewer="ase", **kwargs: dict) -> Any:
        """Visualizes the structure using a number of different viewers.

        Currently, the following viewers are supported:

        - ASE: https://wiki.fysik.dtu.dk/ase/
        - PyVista: https://docs.pyvista.org/
        - A simple homebrew viewer built on matplotlib (not actually
          3D).

        Parameters
        ----------
        viewer
            The viewer to use. By default, the ASE viewer is used.
        **kwargs
            Keyword arguments to pass to the specified viewer.

        Returns
        -------
        handle : Any
            The viewer handle.

        """
        if viewer == "ase":
            return self._view_ase(**kwargs)
        elif viewer == "pyvista":
            return self._view_pyvista(**kwargs)
        elif viewer == "matplotlib":
            return self._view_matplotlib(**kwargs)

    def _view_ase(self, **kwargs) -> Any:
        """Visualizes the structure using the ASE viewer."""
        if ase is None:
            raise ImportError("ASE is not installed.")
        atoms = ase.Atoms(
            symbols=self.kinds,
            positions=self.positions,
            cell=self.cell,
        )
        return ase.visualize.view(atoms, **kwargs)

    def _view_pyvista(self, **kwargs) -> Any:
        """Visualizes the structure using the PyVista viewer."""
        if pv is None:
            raise ImportError("PyVista is not installed.")

        plotter = kwargs.pop("plotter", None)
        if plotter is None:
            plotter = pv.Plotter()

        for kind, position in self.sites:
            color = _jmol_colors[kind]
            size = 15.0 + list(_jmol_colors).index(kind) * 0.25
            plotter.add_points(position, color=color, point_size=size)

        # Generate a mesh of the cell.
        cell = pv.Cube()
        transform = np.eye(4)
        transform[:3, :3] = self.cell.T

        cell.transform(transform)

        plotter.add_mesh(
            cell.extract_feature_edges(),
            style="wireframe",
            color="black",
            line_width=2,
            opacity=0.5,
        )

        return plotter

    def _view_matplotlib(self, **kwargs) -> Axes3D:
        """Visualizes the structure using matplotlib.

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to matplotlib.

        Returns
        -------
        Axes3D
            The matplotlib axes object.

        Notes
        -----
        The plot this produces is sort of wonky as matplotlib does not
        have a proper 3D engine. It can still be useful for quick
        visualizations, though.

        """
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        for kind, position in self.sites:
            color = _jmol_colors[kind]
            size = 50 + list(_jmol_colors).index(kind)
            ax.scatter(*position, c=color, s=size, edgecolors="black")

        for i, j, k in np.ndindex((3, 3, 3)):
            # Base lattice vectors.
            xs_0 = np.array([0, self.cell[i][0]])
            ys_0 = np.array([0, self.cell[i][1]])
            zs_0 = np.array([0, self.cell[i][2]])
            ax.plot(xs_0, ys_0, zs_0, "k--", **kwargs)

            # First order vectors.
            if i == j:
                continue
            xs_1 = xs_0 + [self.cell[i][0], self.cell[j][0]]
            ys_1 = ys_0 + [self.cell[i][1], self.cell[j][1]]
            zs_1 = zs_0 + [self.cell[i][2], self.cell[j][2]]
            ax.plot(xs_1, ys_1, zs_1, "k--", **kwargs)

            # Second order vectors.
            if k == i or k == j:
                continue
            xs_2 = xs_1 + [self.cell[j][0], self.cell[k][0]]
            ys_2 = ys_1 + [self.cell[j][1], self.cell[k][1]]
            zs_2 = zs_1 + [self.cell[j][2], self.cell[k][2]]
            ax.plot(xs_2, ys_2, zs_2, "k--", **kwargs)

        # There is no working equivalent for ax.set_aspect("equal").
        # This is a workaround.
        # https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        return ax
