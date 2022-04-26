""" TODO

"""

import logging
import os

import numpy as np
from ntcad.core.structure import Structure, _symbols

logger = logging.Logger(__name__)

# The minimal potentials necessary.
_minimal_potentials = {
    "K": "_pv",
    "Ca": "_pv",
    "Rb": "_pv",
    "Sr": "_sv",
    "Y": "_sv",
    "Zr": "_sv",
    "Nb": "_pv",
    "Cs": "_sv",
    "Ba": "_sv",
    "Fr": "_sv",
    "Ra": "_sv",
    "Sc": "_sv",
}


# The recommended VASP pseudopotentials that deviate from minimal.
# https://www.vasp.at/wiki/index.php/Available_PAW_potentials
_recommended_potentials = {
    "Li": "_sv",
    "Na": "_pv",
    "K": "_sv",
    "Ca": "_sv",
    "Sc": "_sv",
    "Ti": "_sv",
    "V": "_sv",
    "Cr": "_pv",
    "Mn": "_pv",
    "Ga": "_d",
    "Ge": "_d",
    "Rb": "_sv",
    "Sr": "_sv",
    "Y": "_sv",
    "Zr": "_sv",
    "Nb": "_sv",
    "Mo": "_sv",
    "Tc": "_pv",
    "Ru": "_pv",
    "Rh": "_pv",
    "In": "_d",
    "Sn": "_d",
    "Cs": "_sv",
    "Ba": "_sv",
    "Pr": "_3",
    "Nd": "_3",
    "Pm": "_3",
    "Sm": "_3",
    "Eu": "_2",
    "Gd": "_3",
    "Tb": "_3",
    "Dy": "_3",
    "Ho": "_3",
    "Er": "_3",
    "Tm": "_3",
    "Yb": "_2",
    "Lu": "_3",
    "Hf": "_pv",
    "Ta": "_pv",
    "W": "_pv",
    "Tl": "_d",
    "Pb": "_d",
    "Bi": "_d",
    "Po": "_d",
    "At": "_d",
    "Fr": "_sv",
    "Ra": "_sv",
}


def read_poscar(path: os.PathLike) -> Structure:
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


def write_incar(path: os.PathLike, **kwargs: dict) -> None:
    """Writes an INCAR file at the given path.

    Parameters
    ----------
    path
        Path where to write the INCAR. If any filename is present, the
        filename is replaced with INCAR.
    parameters
        The INCAR tags and their corresponding values that are to be
        written to the file.

    """
    lines = ["INCAR written by ntcad\n"]
    for tag, value in kwargs.items():
        line = tag.upper() + " = "
        if isinstance(value, (list, tuple)):
            line += " ".join(list(map(str, value)))
        else:
            line += str(value)
        lines.append(line + "\n")

    with open(os.path.join(os.path.dirname(path), "INCAR"), "w") as incar:
        incar.writelines(lines)


def write_poscar(path: os.PathLike, structure: Structure) -> None:
    """_summary_

    Parameters
    ----------
    path
        _description_
    structure
        _description_

    """
    lines = ["POSCAR written by ntcad\n", f"{1.0:.5f}\n"]
    for vec in structure.cell:
        lines.append("{:.16f} {:22.16f} {:22.16f}\n".format(*vec))

    kinds, counts = np.unique(structure.kinds, return_counts=True)
    lines.append(" ".join(kinds) + "\n")
    lines.append("  ".join(map(str, counts)) + "\n")

    # TODO allow switch between cartesian and direct. Only Cartesian for
    # now.
    lines.append("Cartesian\n")
    for position in structure.positions:
        lines.append("{:.16f} {:22.16f} {:22.16f}\n".format(*position))

    with open(os.path.join(os.path.dirname(path), "POSCAR"), "w") as poscar:
        poscar.writelines(lines)


def write_potcar(
    path: os.PathLike,
    structure: Structure,
    potentials: dict = None,
    recommended_potentials: bool = False,
):
    """_summary_

    Parameters
    ----------
    path
        _description_
    structure
        _description_
    custom_potentials, optional
        _description_, by default None

    """
    if "VASP_PP_PATH" not in os.environ:
        raise EnvironmentError("Please set the VASP_PP_PATH variable.")

    # TODO: Allow for non-PBE PAW potentials.
    base_path = os.path.join(os.environ["VASP_PP_PATH"], "potpaw_PBE")

    if potentials is None:
        potentials = _minimal_potentials
        if recommended_potentials:
            potentials = _recommended_potentials
    else:
        if recommended_potentials:
            potentials = {**_recommended_potentials, **potentials}
        else:
            potentials = {**_minimal_potentials, **potentials}

    with open(os.path.join(os.path.dirname(path), "POTCAR"), "w") as out_potcar:
        for symbol in np.unique(structure.kinds):
            potcar_folder = symbol + potentials.get(symbol, "")
            potcar_path = os.path.join(base_path, potcar_folder, "POTCAR")
            with open(potcar_path) as in_potcar:
                out_potcar.write(in_potcar.read())


def write_kpoints(path: os.PathLike, kpoints: np.ndarray, shift: np.ndarray = None):
    """_summary_

    Parameters
    ----------
    path
        _description_
    kpoints
        _description_

    """
    if not isinstance(kpoints, np.ndarray):
        kpoints = np.array(kpoints)

    lines = ["KPOINTS written by ntcad\n"]
    # TODO: Maybe properly support line-mode at some point.
    if kpoints.ndim == 1:
        # Monkhorst-Pack grid.
        lines.append("0\n")
        lines.append("Monkhorst-Pack\n")
        lines.append("{:d} {:d} {:d}\n".format(*kpoints))
        if shift is None:
            shift = np.array([0, 0, 0])
        lines.append("{:f} {:f} {:f}\n".format(*shift))
    elif kpoints.ndim == 2:
        # List of k-points.
        lines.append("{:d}\n".format(len(kpoints)))
        lines.append("Reciprocal\n")
        for kpoint in kpoints:
            # Variable length per line in case weights are included.
            lines.append(("{:d}" * len(kpoint)).format(*kpoint) + "\n")
    else:
        raise ValueError(
            "The kpoints should either describe a Monkhorst-Pack grid "
            "or should contain a list of points in reciprocal space."
        )

    with open(os.path.join(os.path.dirname(path), "KPOINTS"), "w") as kpoints:
        kpoints.writelines(lines)
