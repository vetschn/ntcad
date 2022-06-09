"""
This module implements the structure class, defining a configuration of
atoms in a unit cell together with some useful methods.

"""

import os
from typing import Any

import ase.visualize
import matplotlib.pyplot as plt
import ntcad
import numpy as np
from ase import Atoms
from mpl_toolkits.mplot3d.axes3d import Axes3D

# All allowed atomic symbols including a ``None`` / "X" kind.
# Written this way for less clutter (leave me alone).
_symbols = (
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
_numbers = {}
for Z, symbol in enumerate(_symbols):
    _numbers[symbol] = Z

# Useful dtype to represent atomic sites.
_sites_dtype = np.dtype(
    [
        ("kind", np.unicode_, 2),
        ("position", np.float64, 3),
    ]
)

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
    """A configuration of atoms in some unit cell.

    Attributes
    ----------
    sites
        The atomic kinds and positions.
    positions
        The atomic positions
    sites
        The atomic sites and positions in one structure numpy array.

    Methods
    -------
    to_poscar

    to_cif

    from_poscar

    from_cif

    """

    def __init__(
        self,
        kinds: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
        cartesian: bool = True,
        attr: dict = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        kinds
            _description_
        positions
            _description_
        cell
            _description_
        cartesian, optional
            _description_, by default True
        """
        # TODO: Checks whether the atoms are valid, whether the
        # positions are valid, whether the cell is valid, convert
        # negative positions to positive positions in cell, actually
        # check whether cartesian or not.
        self.kinds = np.array(kinds)
        if cartesian:
            self.positions = np.array(positions)
        else:
            self.positions = np.array(positions) @ np.array(cell)
        self.sites = np.array(list(zip(kinds, positions)), dtype=_sites_dtype)
        self.cell = np.array(cell)

        self.attr = None

    def __repr__(self) -> str:
        return f"Structure(\nsites=\n{self.sites},\ncell=\n{self.cell}\n)"

    @property
    def volume(self) -> float:
        """_summary_

        Returns
        -------
            _description_
        """
        a_1, a_2, a_3 = self.cell
        return np.dot(a_1, np.cross(a_2, a_3))

    @property
    def reciprocal_cell(self) -> float:
        """_summary_

        Returns
        -------
            _description_
        """
        return 2 * np.pi * np.transpose(np.linalg.inv(self.cell))

    def view(self, **kwargs) -> Any:
        """_summary_

        Returns
        -------
            _description_
        """
        atoms = Atoms(symbols=self.kinds, positions=self.positions, cell=self.cell)
        return ase.visualize.view(atoms, **kwargs)

    def _mpl_view(self, ax: Axes3D = None) -> Axes3D:
        """_summary_

        Parameters
        ----------
        ax, optional
            _description_, by default None

        Returns
        -------
            _description_
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        for site in self.sites:
            color = _jmol_colors[site["kind"]]
            size = 50 + list(_jmol_colors).index(site["kind"])
            ax.scatter(*site["position"], c=color, s=size, edgecolors="black")

        # NOTE: Not sure if there is a smarter way to draw the cell.
        for i, j, k in np.ndindex((3, 3, 3)):
            # Base lattice vectors.
            xs_0 = np.array([0, self.cell[i][0]])
            ys_0 = np.array([0, self.cell[i][1]])
            zs_0 = np.array([0, self.cell[i][2]])
            ax.plot(xs_0, ys_0, zs_0, "k--")

            # First order vectors.
            if i == j:
                continue
            xs_1 = xs_0 + [self.cell[i][0], self.cell[j][0]]
            ys_1 = ys_0 + [self.cell[i][1], self.cell[j][1]]
            zs_1 = zs_0 + [self.cell[i][2], self.cell[j][2]]
            ax.plot(xs_1, ys_1, zs_1, "k--")

            # Second order vectors.
            if k == i or k == j:
                continue
            xs_2 = xs_1 + [self.cell[j][0], self.cell[k][0]]
            ys_2 = ys_1 + [self.cell[j][1], self.cell[k][1]]
            zs_2 = zs_1 + [self.cell[j][2], self.cell[k][2]]
            ax.plot(xs_2, ys_2, zs_2, "k--")

        # There is no working equivalent for ax.set_aspect("equal").
        # This is a workaround.
        # https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        return ax

    def to_poscar(self, path: os.PathLike) -> None:
        """_summary_

        Parameters
        ----------
        path
            _description_
        """
        ntcad.vasp.io.write_poscar(path, self)

    def to_cif(self, path: os.PathLike) -> None:
        """_summary_

        Parameters
        ----------
        path
            _description_
        """
        #  TODO
        pass

    @classmethod
    def from_poscar(cls, path: os.PathLike) -> "Structure":
        """_summary_

        Parameters
        ----------
        path
            _description_

        Returns
        -------
            _description_
        """
        return vasp.io.read_poscar(path)

    @classmethod
    def from_cif(cls, path: os.PathLike) -> "Structure":
        """_summary_

        Parameters
        ----------
        path
            _description_

        Returns
        -------
            _description_
        """
        # TODO
        pass
