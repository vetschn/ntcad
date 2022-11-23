"""
File I/O
========

This module implements file I/O functions for interfacing with OMEN.

"""

import glob
import os

import numpy as np
from ntcad import Structure, omen
from scipy.sparse import csr_matrix


def read_bin(path: os.PathLike, no_header: bool = False) -> csr_matrix:
    """Parses an OMEN binary sparse matrix file.

    Parameters
    ----------
    path : str or Path
        Path to the binary sparse matrix from OMEN.

    Returns
    -------
    matrix : csr_matrix
        The matrix stored in the file.

    """
    with open(path, "rb") as f:
        bin = np.fromfile(f, dtype=np.double)

    dim, size, one_indexed = tuple(map(int, bin[:3]))
    data = np.reshape(bin[3:], (size, 4))
    row_inds, col_inds, real, imag = data.T

    if one_indexed:
        row_inds, col_inds = row_inds - 1, col_inds - 1

    matrix = csr_matrix(
        (real + 1j * imag, (row_inds, col_inds)),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    return matrix


def write_bin(path: os.PathLike, M: csr_matrix, one_indexed: bool = False) -> None:
    """Writes a sparse matrix to a binary file.

    Parameters
    ----------
    path : str or Path
        Path to write the matrix.
    M : csr_matrix
        The matrix to write.
    one_indexed : bool, optional
        Whether the output matrix should be converted to one-indexed. If
        True, the matrix will be converted to one-indexed before
        writing.

    """
    header = np.array([M.shape[0], M.nnz, float(one_indexed)])

    i, j = np.nonzero(M.sorted_indices())
    vals = np.squeeze(np.array(M[i, j]))

    if one_indexed:
        i, j = i + 1, j + 1

    data = np.stack([i, j, np.real(vals), np.imag(vals)])
    data = data.transpose().flatten()

    with open(path, "wb") as f:
        header.tofile(f)
        data.tofile(f)


def write_P_bin(
    path: os.PathLike, P: np.ndarray, ph_mat_par: dict, layer_matrix: np.ndarray
) -> None:
    """Writes the momentum matrix to a binary file.

    Since this the matrix entries are three dimensional, the matrix
    needs to be reshaped properly before writing.

    Parameters
    ----------
    path : str or Path
        Path to write the momentum matrix.
    P : ndarray
        The momentum matrix to write.
    ph_mat_par : dict
        The material parameters for the momentum matrix.
    layer_matrix : ndarray
        The structure's layer matrix.

    """
    # Extracting some useful information.
    num_orbs = ph_mat_par["orbitals"]

    num_atoms = layer_matrix.shape[0]

    kind_inds = layer_matrix[:, 3].astype(int) - 1
    layer_nn = layer_matrix[:, 4:].astype(int) - 1
    num_nn = layer_nn.shape[-1]

    max_sum_num_orbs = np.sum(num_orbs[kind_inds])

    P_bin = np.zeros(max_sum_num_orbs * (1 + num_nn) * 3 * max(num_orbs))

    k = 0
    for i in range(num_atoms):
        num_orbs_i = num_orbs[kind_inds[i]]
        for j in range(1 + num_nn):
            P_bin[k : k + num_orbs_i * max(num_orbs)] = P[
                i, j, 0, :num_orbs_i, :
            ].T.flatten()
            k += num_orbs_i * max(num_orbs)
            P_bin[k : k + num_orbs_i * max(num_orbs)] = P[
                i, j, 1, :num_orbs_i, :
            ].T.flatten()
            k += num_orbs_i * max(num_orbs)
            P_bin[k : k + num_orbs_i * max(num_orbs)] = P[
                i, j, 2, :num_orbs_i, :
            ].T.flatten()
            k += num_orbs_i * max(num_orbs)

    with open(path, "wb") as f:
        P_bin.tofile(f)


def read_dat(path: os.PathLike) -> np.ndarray:
    """Reads a dat file.

    These are plain text files that can just be read via
    :meth:`numpy.loadtxt`.

    Parameters
    ----------
    path : str or Path
        Path to the dat file.

    Returns
    -------
    data : ndarray
        The data read from the file.

    See Also
    --------
    numpy.loadtxt : The function used to read the file.

    """
    return np.loadtxt(path)


def read_lattice_dat(path: os.PathLike) -> Structure:
    """Reads a lattice.dat file.

    Parameters
    ----------
    path : str or Path
        Path to the lattice.dat file.

    Returns
    -------
    structure : Structure
        The structure read from the file.

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
    """Reads the material parameters file.

    Parameters
    ----------
    path : str or Path
        Path to the file.

    Returns
    -------
    mat_par : dict
        A dictionary containing the information parsed from the file.

    Notes
    -----
    The material parameters file is a plain text file with the following
    format:

    .. code-block:: none

        num_anions num_cations
        Eg Ec_min Ev_max
        [orbitals]
        [masses]

    The file contents are not checked for integrity.

    """
    if not os.path.basename(path).startswith("ph"):
        raise NotImplementedError()

    with open(path, "r") as f:
        lines = f.readlines()

    num_anions, num_cations = tuple(map(int, lines[0].split()))
    Eg, Ec_min, Ev_max = tuple(map(float, lines[1].split()))
    orbitals = np.array(list(map(int, lines[2].split())))
    masses = np.array(list(map(float, lines[3].split())))

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
    """Reads all the coupling matrices from a folder.

    Parameters
    ----------
    path : str or Path
        Path to the folder.

    Returns
    -------
    M : dict
        A dictionary containing the coupling matrices for every
        direction.

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


def read_H_matrices(path: os.PathLike) -> dict:
    """Reads all the Hamiltonian matrices from a folder.

    Parameters
    ----------
    path : str or Path
        Path to the folder.

    Returns
    -------
    H : dict
        A dictionary containing the Hamiltonian matrices for every
        layer.

    """
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")

    bin_files = glob.glob(os.path.join(path, "H_*.bin"))

    H = {}

    for file in bin_files:
        basename = os.path.basename(file)
        num = int(basename[2])
        H[num] = omen.io.read_bin(file)

    return H


def write_e_dat(path: os.PathLike, energies: np.ndarray) -> None:
    """Writes an e.dat file.

    Parameters
    ----------
    path : str or Path

    energies : ndarray
        The energies to write to the file.

    """
    lines = [len(energies)]
    lines.append(("{:22.16f}" * len(energies)).format(*energies))

    if os.path.isfile(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "E_dat"), "w") as e_dat:
        e_dat.writelines(lines)
