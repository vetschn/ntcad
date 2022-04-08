#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" TODO

"""

from matplotlib.pyplot import axis
import numpy as np

import os


def read_hr_dat(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses the contents of a `seedname_hr.dat` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line gives the number of Wigner-Seitz
    grid-points `nrpts`.

    The next block of `nrpts` integers gives the degeneracy of each
    Wigner-Seitz grid point, with 15 entries per line.

    Finally, the `remaining num_wann**2 * nrpts` lines each contain,
    respectively, the components of the vector `R` in terms of the
    lattice vectors {Ai}, the indices m and n, and the real and
    imaginary parts of the Hamiltonian matrix element `H_R_mn` in the WF
    basis.

    Parameters
    ----------
    fn
        Path to `seedname_hr.dat`.

    Returns
    -------
    R, H_R, degen
        The lattice vectors (`nrpts` x 3), Hamiltonian elements
        (`num_wann` x `num_wann` x `nrpts`), and degeneracies (`nrpts`).

    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_wann = int(lines[1])
    nrpts = int(lines[2])
    num_elements = num_wann**2 * nrpts

    degen = np.ndarray([])
    degen_rows = int(np.ceil(nrpts / 15.0))
    for i in range(degen_rows):
        np.append(degen, list(map(int, lines[i + 3].split())))

    R_mn = np.zeros((num_elements, 3))
    H_R_mn = np.zeros((num_elements), dtype=np.complex64)
    for i in range(num_elements):
        entries = lines[3 + degen_rows + i].split()
        R_mn[i, :] = list(map(int, entries[:3]))
        H_R_mn_real, H_R_mn_imag = tuple(map(float, entries[5:]))
        H_R_mn[i] = H_R_mn_real + 1j * H_R_mn_imag

    R = np.unique(R_mn, axis=0)
    H_R = np.reshape(H_R_mn, (num_wann, num_wann, nrpts), order="F")

    return R, H_R, degen


def read_r_dat(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parses the contents of a `seedname_r.dat` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line states the number of `R` vectors `nrpts`.

    Similar to the case of the Hamiltonian matrix above, the remaining
    `num_wann**2 * nrpts` lines each contain, respectively, the
    components of the vector `R` in terms of the lattice vectors {Ai},
    the indices m and n, and the real and imaginary parts of the
    position matrix element in the WF basis.

    Parameters
    ----------
    path
        Path to `seedname_r.dat`.

    Returns
    -------
    R, r_R
        The lattice vectors (`nrpts` x 3), and the position matrix
        elements (`num_wann` x `num_wann` x `nrpts` x 3).

    """

    with open(path, "r") as f:
        lines = f.readlines()

    num_wann = int(lines[1])
    nrpts = int(lines[2])
    num_elements = num_wann**2 * nrpts

    R_mn = np.zeros((num_elements, 3))
    x_R_mn = np.zeros((num_elements), dtype=np.complex64)
    y_R_mn = np.zeros((num_elements), dtype=np.complex64)
    z_R_mn = np.zeros((num_elements), dtype=np.complex64)
    for i in range(num_elements):
        entries = lines[3 + i].split()
        R_mn[i, :] = list(map(int, entries[:3]))
        x_R_mn_real, x_R_mn_imag = tuple(map(float, entries[5:7]))
        y_R_mn_real, y_R_mn_imag = tuple(map(float, entries[7:9]))
        z_R_mn_real, z_R_mn_imag = tuple(map(float, entries[9:]))
        x_R_mn[i] = x_R_mn_real + 1j * x_R_mn_imag
        y_R_mn[i] = y_R_mn_real + 1j * y_R_mn_imag
        z_R_mn[i] = z_R_mn_real + 1j * z_R_mn_imag

    R = np.unique(R_mn, axis=0)
    r_R_mn = np.column_stack((x_R_mn, y_R_mn, z_R_mn))
    r_R = np.reshape(r_R_mn, (num_wann, num_wann, nrpts, 3), order="F")

    return R, r_R
