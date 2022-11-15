"""
This module implements file I/O functions for interfacing with
Wannier90.

Currently supports the following filetypes:

- ``seedname_hr.dat`` (I/O)
- ``seedname_r.dat`` (I)
- ``seedname_band.dat`` (I)
- ``seedname.eig`` (I)
- ``seedname.wout`` (I)

"""

import os
from datetime import datetime

import numpy as np


def read_hr_dat(path: os.PathLike, full: bool = False) -> tuple[np.ndarray, ...]:
    """Parses the contents of a ``seedname_hr.dat`` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line gives the number of Wigner-Seitz
    grid-points ``nrpts``.

    The next block of ``nrpts`` integers gives the degeneracy of each
    Wigner-Seitz grid point, with 15 entries per line.

    Finally, the remaining ``num_wann**2 * nrpts`` lines each contain,
    respectively, the components of the vector ``R`` in terms of the
    lattice vectors ``A_i``, the indices m and n, and the real and imaginary
    parts of the Hamiltonian matrix element ``H_R_mn`` in the WF basis.

    Parameters
    ----------
    path
        Path to ``seedname_hr.dat``.
    full
        Switch determining nature of return value. When it is ``False``
        (the default) just ``r_R`` is returned, when ``True``, the
        degeneracy info and the allowed Wigner-Seitz cell indices are
        also returned.

    Returns
    -------
        The Hamiltonian elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann``), where ``N_i`` correspond to the
        number of Wigner-Seitz cells along the lattice vectors ``A_i``.
        The indices are chose such that (0, 0, 0) actually gets you the
        center Wigner-Seitz cell. Additionally, if ``full`` is ``True``,
        the degeneracy info and the allowed Wigner-Seitz cell indices
        ``Ra`` are also returned.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Strip info from header.
    num_wann = int(lines[1])
    nrpts = int(lines[2])
    num_elements = num_wann**2 * nrpts

    # Read degeneracy info.
    deg = np.ndarray([])
    deg_rows = int(np.ceil(nrpts / 15.0))
    for i in range(deg_rows):
        np.append(deg, list(map(int, lines[i + 3].split())))

    # Preliminary pass to find number of Wigner-Seitz cells in all
    # directions.
    R_mn = np.zeros((num_elements, 3), dtype=np.int8)
    for i in range(num_elements):
        entries = lines[3 + deg_rows + i].split()
        R_mn[i, :] = list(map(int, entries[:3]))

    R_mn_s = np.subtract(R_mn, R_mn.min(axis=0))
    N_1, N_2, N_3 = R_mn_s.max(axis=0) + 1

    # Obtain Hamiltonian elements.
    H_R = np.zeros((N_1, N_2, N_3, num_wann, num_wann), dtype=np.complex64)
    # R_R = np.zeros((N_1, N_2, N_3, 3), dtype=np.int8)
    for i in range(num_elements):
        entries = lines[3 + deg_rows + i].split()
        R_1, R_2, R_3 = tuple(map(int, entries[:3]))
        m, n = tuple(map(int, entries[3:5]))
        H_R_mn_real, H_R_mn_imag = tuple(map(float, entries[5:]))
        H_R[R_1, R_2, R_3, m - 1, n - 1] = H_R_mn_real + 1j * H_R_mn_imag

    if full:
        return H_R, deg, np.unique(R_mn, axis=0)
    return H_R


def read_r_dat(path: os.PathLike, full: bool = False) -> tuple[np.ndarray, ...]:
    """Parses the contents of a ``seedname_r.dat`` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line states the number of ``R`` vectors ``nrpts``.

    Similar to the case of the Hamiltonian matrix above, the remaining
    ``num_wann**2 * nrpts`` lines each contain, respectively, the
    components of the vector ``R`` in terms of the lattice vectors ``A_i``,
    the indices m and n, and the real and imaginary parts of the
    position matrix element in the WF basis.

    Parameters
    ----------
    path
        Path to ``seedname_r.dat``.
    full
        Switch determining nature of return value. When it is ``False``
        (the default) just ``r_R`` is returned, when ``True``, the allowed
        Wigner-Seitz cell indices are also returned.

    Returns
    -------
        The position matrix elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann``
        x ``num_wann`` x 3), where ``N_i`` correspond to the number of
        Wigner-Seitz cells along the lattice vectors ``A_i``. The indices
        are chosen such that (0, 0, 0) actually gets you the center
        Wigner-Seitz cell. Additionally, if ``full`` is ``True``, the
        allowed Wigner-Seitz cell indices are also returned.

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

    R_mn_s = np.subtract(R_mn, R_mn.min(axis=0))
    N_1, N_2, N_3 = R_mn_s.max(axis=0) + 1

    # Obtain position matrix elements.
    r_R = np.zeros((N_1, N_2, N_3, num_wann, num_wann, 3), dtype=np.complex64)
    for i in range(num_elements):
        entries = lines[3 + i].split()
        R_1, R_2, R_3 = list(map(int, entries[:3]))
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
        r_R[R_1, R_2, R_3, m - 1, n - 1, :] = r_R_mn

    if full:
        return r_R, np.unique(R_mn, axis=0)
    return r_R


def read_band_dat(path: os.PathLike) -> np.ndarray:
    """Parses the contents of a ``seedname_band.dat`` file.

    This file contains the raw data for the interpolated band structure.

    Parameters
    ----------
    path
        Path to ``seedname_band.dat``

    Returns
    -------
        The band structure along the ``kpoint_path`` specified in the
        ``seedname.win`` file (``num_wann`` x ``bands_num_points``).

    """
    with open(path, "r") as f:
        contents = f.read()

    sections = contents.split("\n  \n")[:-1]
    num_wann = len(sections)
    bands_num_points = len(sections[0].split("\n"))

    bands = np.zeros((num_wann, bands_num_points))
    for i, section in enumerate(sections):
        lines = section.split("\n")
        for j, line in enumerate(lines):
            bands[i, j] = float(line.split()[-1])

    return bands


def read_band_kpt(path: os.PathLike) -> np.ndarray:
    """Parses the contents of a ``seedname_band.kpt`` file.

    The k-points used for the interpolated band structure, in units of
    the reciprocal lattice vectors. This file can be used to generate a
    comparison band structure from a first-principles code.

    Parameters
    ----------
    path
        Path to ``seedname_band.kpt``

    Returns
    -------
        The k-points used for the interpolated band structure
        (``num_kpoints`` x 3)

    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_kpoints = lines[0].strip()

    kpt = np.zeros((num_kpoints, 3))
    for i, line in enumerate(lines[1:]):
        kpt[i, :] = list(map(float, line.split()))

    return kpt


def read_eig(path: os.PathLike) -> np.ndarray:
    """Parses the contents of a ``seedname.eig`` file.

    The file ``seedname.eig`` contains the Kohn-Sham eigenvalues [eV] at
    each point in the Monkhorst-Pack mesh.

    Each line consist of two integers and a real number. The first
    integer is the band index, the second integer gives the ordinal
    corresponding to the k-point in the list of k-points in
    ``seedname.win``, and the real number is the eigenvalue.

    Parameters
    ----------
    path
        Path to ``seedname.eig``.

    Returns
    -------
        The Kohn-Sham eigenvalues by band number and k-point number
        (``num_bands`` x ``num_kpoints``).

    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_bands, num_kpoints = lines[-1].split()

    eig = np.zeros((num_bands, num_kpoints))
    for line in lines:
        band, kpoint, value = line.split()
        eig[int(band) - 1, int(kpoint) - 1] = float(value)

    return eig


def _parse_wout_header(lines: list[str]) -> dict:
    """Parses the header section of a ``seedname.wout`` file.

    The header provides some basic information about wannier90, the
    authors, the code version and release, and the execution time of the
    current run.

    Parameters
    ----------
    lines
        Lines of the ``seedname.wout`` file.

    Returns
    -------
        Dictionary containing version, version release date and
        Wannier90 run POSIX timestamp.

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
    """Parses the system information section of a ``seedname.wout`` file.

    This section includes real and reciprocal lattice vectors, atomic
    positions, k-points, parameters for job control, disentanglement,
    localization and plotting

    Parameters
    ----------
    lines
        Lines of the ``seedname.wout`` file.

    Returns
    -------
        Dictionary containing the lattice vectors in real and reciprocal
        space, the atomic sites, the utilized k-grid, the number of
        bands and the number of Wannier functions, as well as the
        "parameters" section as a raw string.

    """
    Ai = np.zeros((3, 3))
    Bi = np.zeros((3, 3))
    for ind, line in enumerate(lines):
        if line.strip().startswith("a_"):
            # Lattice Vectors
            entries = line.split()
            i = int(entries[0][-1]) - 1
            Ai[i, :] = list(map(float, entries[1:]))
        elif line.strip().startswith("b_"):
            # Reciprocal-Space Vectors
            entries = line.split()
            i = int(entries[0][-1]) - 1
            Bi[i, :] = list(map(float, entries[1:]))
        elif "Site" in line:
            # Have to save indices here again because there is no way to
            # know how many atomic sites to expect.
            sites_start_ind = ind + 2
        elif "Grid size" in line:
            sites_stop_ind = ind - 6
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
        __, kind, num, Rx, Ry, Rz, __, Fx, Fy, Fz, __ = line.split()
        sites[f"{kind}_{num}"] = {
            "F": np.array(list(map(float, [Fx, Fy, Fz]))),
            "R": np.array(list(map(float, [Rx, Ry, Rz]))),
        }

    system = {
        "Ai": Ai,
        "Bi": Bi,
        "sites": sites,
        "k_grid": np.array(list(map(int, [k_grid_x, k_grid_y, k_grid_z]))),
        "num_wann": int(num_wann),
        "num_bands": int(num_bands),
        "params": "".join(lines[params_ind:-1]),
    }
    return system


def _parse_wout_k_mesh(lines: list) -> dict:
    # TODO
    ...


def _parse_wout_disentangle(lines: list[str]) -> dict:
    """Parses the disentanglement section of a ``seedname.wout`` file.

    Parameters
    ----------
    lines
        Lines of the ``seedname.wout`` file.

    Returns
    -------
        Dictionary containing information on disentanglement windows and
        the final ``Omega_I``.

    """
    for line in lines:
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
    """Parses the wannierisation section of a ``seedname.wout`` file.

    Parameters
    ----------
    lines
        Lines of the ``seedname.wout`` file.

    Returns
    -------
        Dictionary containing the number of iterations, as well as the
        centers and spreads at each iteration.

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
    # TODO
    ...


def _parse_wout_timing(lines: list) -> dict:
    # TODO
    ...


def read_wout(path: os.PathLike) -> dict:
    """Parses the contents of a ``seedname.wout`` file.

    Parameters
    ----------
    path
        Path to ``seedname.wout``.

    Returns
    -------
        A dictionary representing the contents of the ``seedname.wout``
        file from a Wannier90 run.

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
        # TODO: "k_mesh": _parse_wout_k_mesh(sections[2]),
        "disentangle": _parse_wout_disentangle(sections[3]),
        "wannierise": _parse_wout_wannierise(sections[4]),
        # TODO: "plotting": _parse_wout_plotting(sections[5]), todo
        # TODO: "timing": _parse_wout_timing(sections[6]),
    }
    return wout


def write_hr_dat(
    path: os.PathLike, O_R: np.ndarray, deg: np.ndarray = None, Ra: np.ndarray = None
) -> None:
    """Writes an operator to a ``seedname_hr.dat`` file.

    This function is useful to write a momentum operator that should be
    transformed by winterface for use in OMEN for instance.

    Parameters
    ----------
    path
        Path where to write the ``seedname_hr.dat``
    O_R
        The operator elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann``), where ``N_i`` correspond to the
        number of Wigner-Seitz cells along the lattice vectors ``A_i``.
        The indices are such that (0, 0, 0) actually gets you the center
        Wigner-Seitz cell.
    deg
        Degeneracy info to write to file. If None, writes all zeros.
    Ra
        The allowed Wigner-Seitz cell indices. If None, the function
        only writes the operator for Wigner-Seitz cell indices where the
        operator is non-zero.

    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")
    if O_R.shape[-1] != O_R.shape[-2]:
        raise ValueError(f"Operator at R must be square: {O_R.ndim=}")

    lines = [f"hr.dat written by ntcad | {datetime.now()}\n"]

    # Find the allowed Wigner-Seitz cell indices.
    if Ra is None:
        # Midpoint of the Wigner-Seitz cell indices.
        midpoint = np.floor_divide(np.subtract(O_R.shape[:3], 1), 2)
        Ra = np.array([]).reshape(0, 3)
        for Rs in np.ndindex(O_R.shape[:3]):
            R = Rs - midpoint
            if np.any(O_R[tuple(R)]):
                Ra = np.append(Ra, R.reshape(1, 3), axis=0)

    num_wann = O_R.shape[-1]
    nrpts = Ra.shape[0]
    lines.append(str(num_wann) + "\n")
    lines.append(str(nrpts) + "\n")

    # Construct degeneracy lines.
    if deg is None:
        deg = np.zeros(nrpts, dtype=int)
    deg_per_line = 15  # Magic number.
    deg_str = ""
    for i, val in enumerate(deg):
        deg_str += "{:5d}".format(val)
        if ((i + 1) % deg_per_line) == 0:
            deg_str += "\n\\"
    deg_str += "\n"
    deg_lines = deg_str.split("\\")
    lines.extend(deg_lines)

    # Construct the matrix entry lines.
    for R in Ra:
        R_1, R_2, R_3 = tuple(map(int, R))
        for n, m in np.ndindex(O_R.shape[-2:]):
            O_R_mn = O_R[R_1, R_2, R_3, m, n]
            O_R_mn_real, O_R_mn_imag = O_R_mn.real, O_R_mn.imag
            # NOTE: m and n are one-indexed in hr_dat files.
            line = "{:d} {:5d} {:5d} {:5d} {:5d} ".format(R_1, R_2, R_3, m + 1, n + 1)
            line += "{:22.10e} {:22.10e}\n".format(O_R_mn_real, O_R_mn_imag)
            lines.append(line)

    if not path.endswith("_hr.dat"):
        path += "_hr.dat"

    with open(path, "w") as hr_dat:
        hr_dat.writelines(lines)
