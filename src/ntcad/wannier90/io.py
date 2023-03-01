"""
This module implements file I/O functions for interfacing with
Wannier90.

"""

import os
from datetime import datetime
from typing import Any

import numpy as np

from ntcad.__about__ import __version__
from ntcad.structure import Structure
from ntcad.utils import ndrange


def read_hr_dat(path: os.PathLike, return_all: bool = False) -> tuple[np.ndarray, ...]:
    """Parses the contents of a `seedname_hr.dat` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line gives the number of Wigner-Seitz
    grid-points.

    The next block of integers gives the degeneracy of each Wigner-Seitz
    grid point, arranged into 15 values per line.

    Finally, the remaining lines each contain, respectively, the
    components of the Wigner-Seitz cell index, the Wannier center
    indices m and n, and and the real and imaginary parts of the
    Hamiltonian matrix element `HRmn` in the localized basis.

    Parameters
    ----------
    path : os.PathLike
        Path to a `seedname_hr.dat` file.
    return_all : bool, optional
        Whether to return all the data or just the Hamiltonian in the
        localized basis. When `True`, the degeneracies and the
        Wigner-Seitz cell indices are also returned. Defaults to
        `False`.

    Returns
    -------
    hr : np.ndarray
        The Hamiltonian matrix elements in the localized basis.
    degeneracies : np.ndarray, optional
        The degeneracies of the Wigner-Seitz grid points.
    R : np.ndarray, optional
        The Wigner-Seitz cell indices.

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
    R = np.zeros((num_elements, 3), dtype=np.int8)
    for i in range(num_elements):
        entries = lines[3 + deg_rows + i].split()
        R[i, :] = list(map(int, entries[:3]))

    Rs = np.subtract(R, R.min(axis=0))
    N1, N2, N3 = Rs.max(axis=0) + 1

    # Obtain Hamiltonian elements.
    hR = np.zeros((N1, N2, N3, num_wann, num_wann), dtype=np.complex64)
    for i in range(num_elements):
        entries = lines[3 + deg_rows + i].split()
        R1, R2, R3 = map(int, entries[:3])
        m, n = map(int, entries[3:5])
        hR_mn_real, hR_mn_imag = map(float, entries[5:])
        hR[R1, R2, R3, m - 1, n - 1] = hR_mn_real + 1j * hR_mn_imag

    if return_all:
        return hR, deg, np.unique(R, axis=0)
    return hR


def read_r_dat(path: os.PathLike, full: bool = False) -> tuple[np.ndarray, ...]:
    """Parses the contents of a ``seedname_r.dat`` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    num_wann. The third line states the number of ``R`` vectors
    ``nrpts``.

    Similar to the case of the Hamiltonian matrix, the remaining
    ``num_wann**2 * nrpts`` lines each contain, respectively, the
    components of the vector ``R`` in terms of the lattice vectors
    ``Ai``, the indices m and n, and the real and imaginary parts of the
    position matrix element in the WF basis.

    Parameters
    ----------
    path : os.PathLike
        Path to ``seedname_r.dat``.
    full : bool, optional
        Switch determining nature of return value. When it is ``False``
        (the default) just ``r_R`` is returned, when ``True``, the
        allowed Wigner-Seitz cell indices are also returned.

    Returns
    -------
    rR : np.ndarray
        The position matrix elements in the WF basis.
    R : np.ndarray, optional
        The allowed Wigner-Seitz cell indices.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_wann = int(lines[1])
    nrpts = int(lines[2])
    num_elements = num_wann**2 * nrpts

    # Preliminary pass to find number of Wigner-Seitz cells.
    R = np.zeros((num_elements, 3), dtype=np.int8)
    for i in range(num_elements):
        entries = lines[3 + i].split()
        R[i, :] = list(map(int, entries[:3]))

    Rs = np.subtract(R, R.min(axis=0))
    N1, N2, N3 = Rs.max(axis=0) + 1

    # Obtain position matrix elements.
    rR = np.zeros((N1, N2, N3, num_wann, num_wann, 3), dtype=np.complex64)
    for i in range(num_elements):
        entries = lines[3 + i].split()
        R1, R2, R3 = map(int, entries[:3])
        m, n = map(int, entries[3:5])
        xR_mn_real, xR_mn_imag = map(float, entries[5:7])
        yR_mn_real, yR_mn_imag = map(float, entries[7:9])
        zR_mn_real, zR_mn_imag = map(float, entries[9:])
        rR_mn = np.array(
            [
                xR_mn_real + 1j * xR_mn_imag,
                yR_mn_real + 1j * yR_mn_imag,
                zR_mn_real + 1j * zR_mn_imag,
            ]
        )
        rR[R1, R2, R3, m - 1, n - 1, :] = rR_mn

    if full:
        return rR, np.unique(R, axis=0)
    return rR


def read_band_dat(path: os.PathLike) -> np.ndarray:
    """Parses the contents of a ``seedname_band.dat`` file.

    This file contains the raw data for the interpolated band structure.

    Parameters
    ----------
    path : os.PathLike
        Path to ``seedname_band.dat``

    Returns
    -------
    bands : np.ndarray
        The interpolated band structure.

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
    path : os.PathLike
        Path to ``seedname_band.kpt``

    Returns
    -------
    kpoints : np.ndarray
        The k-points used for the interpolated band structure.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_kpoints = lines[0].strip()

    kpoints = np.zeros((num_kpoints, 3))
    for i, line in enumerate(lines[1:]):
        kpoints[i, :] = list(map(float, line.split()))

    return kpoints


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
    path : os.PathLike
        Path to ``seedname.eig``.

    Returns
    -------
    eigs : np.ndarray
        The Kohn-Sham eigenvalues by band and k-point.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_bands, num_kpoints = lines[-1].split()

    eigs = np.zeros((num_bands, num_kpoints))
    for line in lines:
        band, kpoint, value = line.split()
        eigs[int(band) - 1, int(kpoint) - 1] = float(value)

    return eigs


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
    path : os.PathLike
        Path to ``seedname.wout``.

    Returns
    -------
    wout : dict
        Dictionary containing the parsed contents of the ``seedname.wout``
        file.

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


def _parse_xsf_crystal(lines: list[str]) -> Structure:
    """Parses the CRYSTAL section of a ``seedname.xsf`` file."""
    for ind, line in enumerate(lines):
        if line.startswith("PRIMVEC"):
            cell = np.zeros((3, 3))
            for i in range(3):
                cell[i] = list(map(float, lines[ind + 1 + i].split()))

        elif line.startswith("PRIMCOORD"):
            num_sites = int(lines[ind + 1].split()[0])
            kinds = []
            positions = np.zeros((num_sites, 3))
            for i in range(num_sites):
                line = lines[ind + 2 + i].split()
                kinds.append(line[0])
                positions[i] = list(map(float, line[1:]))
            kinds = np.array(kinds)
    return Structure(kinds, positions, cell)


def _parse_xsf_datagrid(lines: list[str]) -> np.ndarray:
    """Parses the DATAGRID_xD section of a ``seedname.xsf`` file."""
    lines = [l.strip() for l in lines if not l.startswith("END_")]

    dim = int(lines[0].split("_")[2][0])
    shape = tuple(map(int, lines[1].strip().split()))
    origin = np.array(list(map(float, lines[2].strip().split())))
    cell = np.zeros((dim, 3))
    for i in range(dim):
        cell[i] = list(map(float, lines[3 + i].strip().split()))

    data = np.array(list(map(float, " ".join(lines[3 + dim :]).split())))
    datagrid = {
        "shape": shape,
        "origin": origin,
        "cell": cell,
        "data": data,
    }
    return datagrid


def read_xsf(path: os.PathLike, data_only: bool = False) -> Any:
    """Parses the contents of a ``seedname.xsf`` file.

    .. note::

        Molecules and animations are not supported.

    Parameters
    ----------
    path : os.PathLike
        Path to ``seedname.xsf``.
    data_only : bool, optional
        If ``True``, only the data is returned. Otherwise, the data is
        returned as a structure attribute. Default is ``False``.

    Returns
    -------
    Structure
        Structure object containing the parsed contents of the
        ``seedname.xsf`` file.

    Notes
    -----
    The ``seedname.xsf`` file is a text file containing the coordinates
    of the atoms in the unit cell, as well as the unit cell vectors.

    See the `official documentation
    <http://www.xcrysden.org/doc/XSF.html>`_ for more information.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Remove comment lines.
    lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]

    supported = {"CRYSTAL", "BEGIN_"}  # "END" does not need to be checked.
    unsupported = {"ATOMS", "MOLECULE", "POLYMER", "SLAB", "ANIMSTEPS"}
    keywords = supported | unsupported

    section_inds = []
    for ind, line in enumerate(lines):
        if any(name in line for name in keywords):
            section_inds += [ind]
        if any(name in line for name in unsupported):
            raise NotImplementedError(f"{line.strip()} not supported.")

    sections = [lines[i:j] for i, j in zip(section_inds, section_inds[1:] + [None])]

    attr = {}
    for ind, section in enumerate(sections):
        if "CRYSTAL" in section[0]:
            structure = _parse_xsf_crystal(section)
        elif "BEGIN_BLOCK_DATAGRID" in section[0]:
            key = section[1].strip()
            datagrid = _parse_xsf_datagrid(sections[ind + 1])
            attr[key] = datagrid

    if data_only:
        return attr

    structure.attr = attr
    return structure


def write_hr_dat(
    path: os.PathLike, O_R: np.ndarray, deg: np.ndarray = None, Ra: np.ndarray = None
) -> None:
    """Writes an operator to a ``seedname_hr.dat`` file.

    This function is useful to write a momentum operator that should be
    transformed by winterface for use in OMEN for instance.

    Parameters
    ----------
    path : os.PathLike
        Path where to write the ``seedname_hr.dat``
    O_R : np.ndarray
        The operator elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann``), where ``N_i`` correspond to the
        number of Wigner-Seitz cells along the lattice vectors ``A_i``.
        The indices are such that (0, 0, 0) actually gets you the center
        Wigner-Seitz cell.
    deg : np.ndarray, optional
        Degeneracy info to write to file. If None, writes all zeros.
    Ra : np.ndarray, optional
        The allowed Wigner-Seitz cell indices. If None, the function
        only writes the operator for Wigner-Seitz cell indices where the
        operator is non-zero.

    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")
    if O_R.shape[-1] != O_R.shape[-2]:
        raise ValueError(f"Operator at R must be square: {O_R.ndim=}")

    lines = [f"hr.dat written by ntcad v{__version__} | {datetime.now()}\n"]

    # Find the allowed Wigner-Seitz cell indices.
    if Ra is None:
        Ra = np.array([]).reshape(0, 3)
        for R in ndrange(O_R.shape[:3], centered=True):
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
        R1, R2, R3 = tuple(map(int, R))
        for n, m in np.ndindex(O_R.shape[-2:]):
            O_R_mn = O_R[R1, R2, R3, m, n]
            O_R_mn_real, O_R_mn_imag = O_R_mn.real, O_R_mn.imag
            # NOTE: m and n are one-indexed in hr_dat files.
            line = "{:d} {:5d} {:5d} {:5d} {:5d} ".format(R1, R2, R3, m + 1, n + 1)
            line += "{:22.10e} {:22.10e}\n".format(O_R_mn_real, O_R_mn_imag)
            lines.append(line)

    if not path.endswith("_hr.dat"):
        path += "_hr.dat"

    with open(path, "w") as hr_dat:
        hr_dat.writelines(lines)
