""" TODO

"""

import os
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from ntcad.core.structure import Structure

from scipy import constants

m_u = constants["atomic mass constant"]


def read_bin(path: os.PathLike) -> csr_matrix:
    """Parses an OMEN binary sparse matrix file.

    Parameters
    ----------
    path
        Path to the binary sparse matrix.

    Returns
    -------
    csr_matrix
        The matrix stored in the file as ``scipy.sparse.csr_matrix``.

    """
    with open(path, "rb") as f:
        bin = np.fromfile(f, dtype=np.double)

    dim, size, one_indexed = tuple(map(int, bin[:3]))
    data = bin[3:].reshape(size, 4)
    row_ind, col_ind, real, imag = data.T

    if one_indexed:
        row_ind, col_ind = row_ind - 1, col_ind - 1

    matrix = csr_matrix(
        (real + 1j * imag, (row_ind, col_ind)),
        shape=(dim, dim),
        dtype=np.complex64,
    )
    return matrix


def read_layer_matrix(path: os.PathLike) -> Any:
    """_summary_

    Parameters
    ----------
    path
        _description_

    Returns
    -------
        _description_

    """
    layer_matrix = np.loadtxt(path)
    coords = layer_matrix[:, :3]
    kinds = layer_matrix[:, 3]
    nn = layer_matrix[:, 4:]


def read_lattice_dat(path: os.PathLike) -> Any:
    """_summary_

    Parameters
    ----------
    path
        _description_

    Returns
    -------
        _description_

    """
    with open(path, "r") as f:
        lines = f.readlines()


def read_mat_par(path: os.PathLike) -> dict:
    """_summary_

    Parameters
    ----------
    path
        _description_

    Returns
    -------
    mat_par
        _description_

    """
    if not os.path.basename(path).startswith("ph"):
        raise NotImplementedError()

    with open(path, "r") as f:
        lines = f.readlines()

    num_anions, num_cations = tuple(map(int, lines[0].split()))
    Eg, Ec_min, Ev_max = tuple(map(float, lines[1].split()))
    num_orbitals = np.array(list(map(int, lines[2].split())))
    atomic_masses = np.array(list(map(float, lines[3].split()))) * m_u

    mat_par = {
        "num_anions": num_anions,
        "num_cations": num_cations,
        "Eg": Eg,
        "Ec_min": Ec_min,
        "Ev_max": Ev_max,
        "num_orbitals": num_orbitals,
        "atomic_masses": atomic_masses,
    }

    return mat_par
