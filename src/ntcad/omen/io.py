"""
This module implements file I/O functions for interfacing with OMEN.

"""
import glob
import os

import numpy as np
from ntcad import omen
from ntcad.core.structure import Structure
from scipy import constants
from scipy.sparse import csr_matrix

m_u, *__ = constants.physical_constants["atomic mass constant"]


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


def read_dat(path: os.PathLike) -> np.ndarray:
    """TODO

    Parameters
    ----------
    path
        _description_

    Returns
    -------
        _description_

    """
    return np.loadtxt(path)


def read_lattice_dat(path: os.PathLike) -> Structure:
    """TODO

    Parameters
    ----------
    path
        _description_

    Returns
    -------
        _description_

    """
    with open(path) as f:
        lines = f.readlines()
    num_sites, *__ = tuple(map(int, lines[0].strip().split()))

    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = list(map(float, lines[2 + i].split()))

    kinds = np.zeros(num_sites, dtype=(np.unicode_, 2))
    positions = np.zeros((num_sites, 3))
    for i in range(num_sites):
        kinds[i] = lines[6 + i].split()[0]
        positions[i] = list(map(float, lines[6 + i].split()[1:]))

    return Structure(kinds, positions, cell, cartesian=True)


def read_mat_par(path: os.PathLike) -> dict:
    """Reads the material parameters.

    Parameters
    ----------
    path
        Path to a ``mat_par`` file.

    Returns
    -------
    mat_par
        A dictionary containing the information parsed from the
        ``mat_par`` file.

    """
    if not os.path.basename(path).startswith("ph"):
        raise NotImplementedError()

    with open(path, "r") as f:
        lines = f.readlines()

    num_anions, num_cations = tuple(map(int, lines[0].split()))
    Eg, Ec_min, Ev_max = tuple(map(float, lines[1].split()))
    orbitals = np.array(list(map(int, lines[2].split())))
    masses = np.array(list(map(float, lines[3].split()))) * m_u

    mat_par = {
        "num_anions": num_anions,
        "num_cations": num_cations,
        "Eg": Eg,
        "Ec_min": Ec_min,
        "Ev_max": Ev_max,
        "orbitals": orbitals,
        "masses": masses,
    }

    return mat_par


def read_M_matrices(path: os.PathLike) -> dict:
    """_summary_

    Parameters
    ----------
    path
        _description_

    Returns
    -------
        _description_
    """
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")

    bin_files = glob.glob(os.path.join(path, "M*_H_*.bin"))

    M = {"x": {}, "y": {}, "z": {}}

    for file in bin_files:
        basename = os.path.basename(file)
        dim = basename[1]
        num = int(basename[5])
        M[dim][num] = omen.io.read_bin(file)
    return M


def write_e_dat(path: os.PathLike, energies: np.ndarray) -> None:
    """_summary_

    Parameters
    ----------
    path
        _description_
    energies
        _description_
    """
    lines = [len(energies)]
    lines.append(("{:22.16f}" * len(energies)).format(*energies))

    if os.path.isfile(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "E_dat"), "w") as e_dat:
        e_dat.writelines(lines)
