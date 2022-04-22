""" TODO

"""

import os

import numpy as np
from typing import Any, Dict
from ntcad.core.structure import Structure


def write_incar(path: os.PathLike, parameters: Dict[str, Any]) -> None:
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
    lines = ["INCAR written by ntcad"]
    for tag, value in parameters.items():
        line = tag.upper() + " = "
        if isinstance(value, list):
            line += " ".join(list(map(str, value)))
        else:
            line += str(value)
        lines.append(line)

    with open(os.path.join(os.path.dirname(path)), "w") as incar:
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
    lines = ["POSCAR written by ntcad"]

    # TODO: Do something.

    with open(os.path.join(os.path.dirname(path)), "w") as incar:
        incar.writelines(lines)


def write_kpoints(path: os.PathLike, kpoints: np.ndarray):
    """_summary_

    Parameters
    ----------
    path
        _description_
    kpoints
        _description_
    """
    lines = ["POSCAR written by ntcad"]

    # TODO: Do something.

    with open(os.path.join(os.path.dirname(path)), "w") as incar:
        incar.writelines(lines)


def write_potcar(path: os.PathLike, structure: Structure):
    """_summary_

    Parameters
    ----------
    path
        _description_
    structure
        _description_
    """
    # TODO: Do something.
