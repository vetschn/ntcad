"""
This module implements file I/O functions for interfacing with CP2K.

"""

import os
import re

import numpy as np


def _parse_param_set(lines: list[str]) -> dict:
    """Parses a GTO parameter set in the CP2K format.

    Parameters
    ----------
    lines : list of str
        The lines specifying the set of parameters.

    Returns
    -------
    parameter_set : dict
        A dictionary containing the parsed parameters.

    """
    n, l_min, l_max, num_exp, *num_shells = map(int, lines[0].split())
    num_con = dict(zip(range(l_min, l_max + 1), num_shells))
    params = np.genfromtxt(lines[1:])
    exponents = params[:, 0]
    con_coefficients = {
        l: params[:, 1 + i : 1 + i + num_con[l]]
        for i, l in enumerate(range(l_min, l_max + 1))
    }

    param_set = {
        "n": n,
        "l_min": l_min,
        "l_max": l_max,
        "num_exponents": num_exp,
        "num_contractions": num_con,
        "a": exponents,
        "c": con_coefficients,
    }

    return param_set


def read_basis_set(lines: list[str]) -> dict:
    """Parses a basis set in the CP2K format.

    Parameters
    ----------
    lines : list of str
        The lines containing the basis set information.

    Returns
    -------
    basis_set : dict
        A dictionary containing the parsed basis set.

    """
    element, *name = lines[0].split()
    num_param_sets = int(lines[1])

    # Parse each parameter set.
    param_sets = []
    offset = 2
    for __ in range(num_param_sets):
        num_exponents = int(lines[offset].split()[3])
        param_sets.append(_parse_param_set(lines[offset : offset + num_exponents + 1]))
        offset += num_exponents + 1

    basis_set = {
        "element": element,
        "name": " ".join(name),
        "num_param_sets": num_param_sets,
        "param_sets": param_sets,
    }

    return basis_set


def read_basis_set_database(file: os.PathLike) -> list[dict]:
    """Parses a basis set file in the CP2K format.

    These are files like `BASIS_MOLOPT`, `BASIS_MOLOPT_UCL`, etc.

    Parameters
    ----------
    file : str or Path
        Path to the file.

    Returns
    -------
    basis_sets : list of dict
        A list of dictionaries containing the parsed basis sets.

    """
    with open(file, "r") as f:
        lines = f.readlines()

    # Remove comments and empty lines.
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != "" and line[0] != "#"]

    # Split everything per basis set.
    basis_sets = re.split("(^[A-Z])", "\n".join(lines), flags=re.MULTILINE)
    basis_sets = [el + params for el, params in zip(basis_sets[1::2], basis_sets[2::2])]

    # Parse each basis set.
    return [read_basis_set(basis_set.splitlines()) for basis_set in basis_sets]
