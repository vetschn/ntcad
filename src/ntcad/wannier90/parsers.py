#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" TODO: Docstrings.

"""

from datetime import datetime
from matplotlib.pyplot import axis

import numpy as np


def read_hr_dat(path: str) -> np.ndarray:
    """Parses the contents of a `seedname_hr.dat` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line gives the number of Wigner-Seitz
    grid-points `nrpts`.

    The next block of `nrpts` integers gives the degeneracy of each
    Wigner-Seitz grid point, with 15 entries per line.

    Finally, the remaining `num_wann**2 * nrpts` lines each contain,
    respectively, the components of the vector `R` in terms of the
    lattice vectors A_i, the indices m and n, and the real and imaginary
    parts of the Hamiltonian matrix element `H_R_mn` in the WF basis.

    Parameters
    ----------
    path
        Path to `seedname_hr.dat`.

    Returns
    -------
    H_R, degen
        The Hamiltonian elements (`N_1` x `N_2` x `N_3` x `num_wann` x
        `num_wann`), and the degeneracies (`nrpts`), where `N_i`
        correspond to the number of Wigner-Seitz cells along the lattice
        vectors `A_i`.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Strip info from header.
    num_wann = int(lines[1])
    nrpts = int(lines[2])
    num_elements = num_wann**2 * nrpts

    # Read degeneracy info.
    degen = np.ndarray([])
    degen_rows = int(np.ceil(nrpts / 15.0))
    for i in range(degen_rows):
        np.append(degen, list(map(int, lines[i + 3].split())))

    # Preliminary pass to find number of Wigner-Seitz cells.
    R_mn = np.zeros((num_elements, 3), dtype=np.int8)
    for i in range(num_elements):
        entries = lines[3 + degen_rows + i].split()
        R_mn[i, :] = list(map(int, entries[:3]))

    # Shift the `R` vector such that it can be used to index H_R.
    R_mn_index = np.subtract(R_mn, R_mn.min(axis=0))
    R_1, R_2, R_3 = R_mn_index.T
    N_1, N_2, N_3 = R_mn_index.max(axis=0) + 1

    # Obtain Hamiltonian elements.
    H_R = np.zeros((N_1, N_2, N_3, num_wann, num_wann), dtype=np.complex64)
    for i in range(num_elements):
        entries = lines[3 + degen_rows + i].split()
        m, n = tuple(map(int, entries[3:5]))
        H_R_mn_real, H_R_mn_imag = tuple(map(float, entries[5:]))
        H_R[R_1[i], R_2[i], R_3[i], m - 1, n - 1] = H_R_mn_real + 1j * H_R_mn_imag

    return H_R, degen


def read_r_dat(path: str) -> np.ndarray:
    """Parses the contents of a `seedname_r.dat` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line states the number of `R` vectors `nrpts`.

    Similar to the case of the Hamiltonian matrix above, the remaining
    `num_wann**2 * nrpts` lines each contain, respectively, the
    components of the vector `R` in terms of the lattice vectors `A_i`,
    the indices m and n, and the real and imaginary parts of the
    position matrix element in the WF basis.

    Parameters
    ----------
    path
        Path to `seedname_r.dat`.

    Returns
    -------
    r_R
        The position matrix elements (`N_1` x `N_2` x `N_3` x `num_wann`
        x `num_wann`x 3), where `N_i` correspond to the number of
        Wigner-Seitz cells along the lattice vectors `A_i`.

    """

    with open(path, "r") as f:
        lines = f.readlines()

    num_wann = int(lines[1])
    nrpts = int(lines[2])
    num_elements = num_wann**2 * nrpts

    # Preliminary pass to find number of Wigner-Seitz cells.
    R_mn = np.zeros((num_elements, 3), dtype=np.int8)
    for i in range(num_elements):
        entries = lines[3 + i].split()
        R_mn[i, :] = list(map(int, entries[:3]))

    # Shift the `R` vector such that it can be used to index H_R.
    R_mn_index = np.subtract(R_mn, R_mn.min(axis=0))
    R_1, R_2, R_3 = R_mn_index.T
    N_1, N_2, N_3 = R_mn_index.max(axis=0) + 1

    # Obtain position matrix elements.
    r_R = np.zeros((N_1, N_2, N_3, num_wann, num_wann, 3), dtype=np.complex64)
    for i in range(num_elements):
        entries = lines[3 + i].split()
        m, n = tuple(map(int, entries[3:5]))
        x_R_mn_real, x_R_mn_imag = tuple(map(float, entries[5:7]))
        y_R_mn_real, y_R_mn_imag = tuple(map(float, entries[7:9]))
        z_R_mn_real, z_R_mn_imag = tuple(map(float, entries[9:]))
        r_R_mn = np.array(
            [
                x_R_mn_real + 1j * x_R_mn_imag,
                y_R_mn_real + 1j * y_R_mn_imag,
                z_R_mn_real + 1j * z_R_mn_imag,
            ]
        )
        r_R[R_1[i], R_2[i], R_3[i], m - 1, n - 1, :] = r_R_mn

    return r_R


def _parse_wout_header(lines: list[str]) -> dict:
    """Parses the header section of a `seedname.wout` file.

    The header provides some basic information about wannier90, the
    authors, the code version and release, and the execution time of the
    current run

    Parameters
    ----------
    lines
        Lines of the `seedname.wout` file.

    Returns
    -------
        _description_
    """
    for line in lines:
        if "Release" in line:
            *__, version, day, month, year, __ = line.split()
        elif "Execution started on" in line:
            *__, date, __, time, __ = line.split()
            execution_dt = datetime.strptime(f"{date} {time}", "%d%b%Y %H:%M:%S")

    header = {
        "version": version,
        "release_date": f"{day} {month} {year}",
        "timestamp": execution_dt.timestamp(),
    }

    return header


def _parse_wout_system(lines: list[str]) -> dict:
    """Parses the system information section of a `seedname.wout` file.

    This section includes real and reciprocal lattice vectors, atomic
    positions, k-points, parameters for job control, disentanglement,
    localisation and plotting

    Parameters
    ----------
    lines
        Lines of the `seedname.wout` file.

    Returns
    -------
        _description_

    """
    A_i = np.zeros((3, 3))
    B_i = np.zeros((3, 3))
    for ind, line in enumerate(lines):
        if line.strip().startswith("a_"):
            # Lattice Vectors
            entries = line.split()
            i = int(entries[0][-1]) - 1
            A_i[i, :] = list(map(float, entries[1:]))
        elif line.strip().startswith("b_"):
            # Reciprocal-Space Vectors
            entries = line.split()
            i = int(entries[0][-1]) - 1
            B_i[i, :] = list(map(float, entries[1:]))
        elif "Site" in line:
            # Have to save indices here again because there is no way to
            # know how many atomic sites to expect.
            sites_start_ind = ind + 2
        elif "Grid size" in line:
            sites_stop_ind = ind - 5
            line = line.replace("x", " ")
            __, __, __, k_grid_x, k_grid_y, k_grid_z, *__ = line.split()
        elif "MAIN" in line:
            params_ind = ind
        elif "Number of Wannier Functions" in line:
            *__, num_wann, __ = line.split()
        elif "Number of input Bloch states" in line:
            *__, num_bands, __ = line.split()

    sites_lines = lines[sites_start_ind:sites_stop_ind]
    sites = {}
    for line in sites_lines:
        __, kind, num, R_x, R_y, R_z, __, F_x, F_y, F_z, __ = line.split()
        sites[f"{kind}_{num}"] = {
            "F": np.array(list(map(float, [F_x, F_y, F_z]))),
            "R": np.array(list(map(float, [R_x, R_y, R_z]))),
        }

    system = {
        "A_i": A_i,
        "B_i": B_i,
        "sites": sites,
        "k_grid": np.array(list(map(int, [k_grid_x, k_grid_y, k_grid_z]))),
        "num_wann": int(num_wann),
        "num_bands": int(num_bands),
        "params": "".join(lines[params_ind:-1]),
    }
    return system


def _parse_wout_k_mesh(lines: list[str]) -> dict:
    """Parses the k-mesh section of a `seedname.wout` file.

    TODO

    This part of the output files provides information on the b-vectors
    and weights chosen.

    Parameters
    ----------
    lines
        Lines of the `seedname.wout` file.

    Returns
    -------
        _description_

    """
    pass


def _parse_wout_disentangle(lines: list[str]) -> dict:
    """Parses the disentanglement section of a `seedname.wout` file.

    Parameters
    ----------
    lines
        Lines of the `seedname.wout` file.

    Returns
    -------
        _description_
    """

    for ind, line in enumerate(lines):
        if "Outer:" in line:
            __, __, dis_win_min, __, dis_win_max, *__ = line.split()
        elif "Inner:" in line:
            __, __, dis_froz_min, __, dis_froz_max, *__ = line.split()
        elif "Final Omega_I" in line:
            Omega_I = line.split()[2]

    disentangle = {
        "dis_win_min": float(dis_win_min),
        "dis_win_max": float(dis_win_max),
        "dis_froz_min": float(dis_froz_min),
        "dis_froz_max": float(dis_froz_max),
        "Omega_I": float(Omega_I),
    }
    return disentangle


def _parse_wout_wannierise(lines: list[str]) -> dict:
    """Parses the wannierisation section of a `seedname.wout` file.

    Parameters
    ----------
    lines
        Lines of the `seedname.wout` file.

    Returns
    -------
        _description_
    """
    wann_inds = []
    centers = []
    spreads = []
    for line in lines:
        if "WF centre and spread" in line:
            line = line.replace(",", " ")
            *__, wann_ind, __, center_x, center_y, center_z, __, spread = line.split()
            wann_inds.append(int(wann_ind))
            centers.append(list(map(float, [center_x, center_y, center_z])))
            spreads.append(float(spread))

    num_iter = wann_inds.count(1)
    num_wann = max(wann_inds)

    centers = np.reshape(centers, (num_iter, num_wann, 3))
    spreads = np.reshape(spreads, (num_iter, num_wann))

    wannierise = {
        "num_iter": num_iter,
        "num_wann": num_wann,
        "centers": centers,
        "spreads": spreads,
    }

    return wannierise


def _parse_wout_plotting(lines: list) -> dict:
    """_summary_

    Parameters
    ----------
    ind
        _description_
    lines
        _description_

    Returns
    -------
        _description_
    """
    pass


def _parse_wout_timing(lines: list[str]) -> dict:
    """Parses the summary timings section of a `seedname.wout` file.

    Parameters
    ----------
    ind
        Index at the section start.
    lines
        Lines of the `seedname.wout` file.

    Returns
    -------
        _description_
    """
    return {}


def read_wout(path: str) -> dict:
    """Parses the contents of a `seedname.wout` file.

    Parameters
    ----------
    path
        Path to `seedname.wout`.

    Returns
    -------
        _description_
    """

    with open(path, "r") as f:
        lines = f.readlines()

    assert "All done" in "".join(lines), "Run incomplete."

    sections_names = (
        "SYSTEM",
        "K-MESH",
        "DISENTANGLE",
        "WANNIERISE",
        "PLOTTING",
        "TIMING INFORMATION",
    )
    section_inds = [0]
    for ind, line in enumerate(lines):
        if any(name in line for name in sections_names) and "---" in lines[ind + 1]:
            section_inds += [ind]

    sections = [lines[i:j] for i, j in zip(section_inds, section_inds[1:] + [None])]

    wout = {
        "header": _parse_wout_header(sections[0]),
        "system": _parse_wout_system(sections[1]),
        # "k_mesh": _parse_wout_k_mesh(sections[2]),
        "disentangle": _parse_wout_disentangle(sections[3]),
        "wannierise": _parse_wout_wannierise(sections[4]),
        # "plotting": _parse_wout_plotting(sections[5]),
        # "timing": _parse_wout_timing(sections[6]),
    }

    return wout
