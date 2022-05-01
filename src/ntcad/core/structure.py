"""
This module implements the structure class, defining a configuration of
atoms in a unit cell together with some useful methods.

"""

import os

import numpy as np

import ntcad

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
        ("kinds", np.unicode_, 2),
        ("positions", np.float64, 3),
    ]
)


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
        self.positions = np.array(positions)
        self.sites = np.array(list(zip(kinds, positions)), dtype=_sites_dtype)
        self.cell = np.array(cell)

        self.attr = None

    def __str__(self) -> str:
        """_summary_

        Returns
        -------
            _description_
        """
        #  TODO
        pass

    def to_poscar(self, path: os.PathLike) -> None:
        """_summary_

        Parameters
        ----------
        path
            _description_
        """
        ntcad.io.vasp.write_poscar(path, self)

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
        # TODO
        pass

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
