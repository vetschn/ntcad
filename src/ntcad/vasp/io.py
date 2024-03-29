"""
This module implements file I/O functions for interfacing with VASP.

"""

import ast
import json
import os
from datetime import datetime

import numpy as np
import xmltodict

from ntcad.__about__ import __version__
from ntcad.structure import Structure

# The minimal pseudopotentials necessary.
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

# --- Input ------------------------------------------------------------


def read_incar(path: os.PathLike) -> dict:
    """Reads the INCAR tags from a given file.

    Parameters
    ----------
    path : str or Path
        Path to the INCAR file to read.

    Returns
    -------
    incar_tags : dict
        A dictionary containing the parsed INCAR tags.

    See Also
    --------
    write_incar : Write INCAR tags in VASP format.

    Notes
    -----
    The INCAR file format is described in the `VASP wiki
    <https://www.vasp.at/wiki/index.php/INCAR>`_. The ``read_incar``
    function parses the file and returns a dictionary of the tags and
    their values. The values are converted to the appropriate Python
    types using :meth:`ast.literal_eval`.

    The INCAR tags are not checked for their validity against the ones
    in the VASP manual.

    This function may run into problems with multi-line values, for
    example the ``WANNIER90_WIN`` tag.

    """
    with open("./INCAR", "r") as f:
        lines = f.readlines()

    # Get rid of any comments or other nonsense.
    lines = [line for line in lines if "=" in line]
    lines = [line.split("#")[0].split("!")[0] for line in lines]

    # Split apart multiple tags on a single line.
    lines = [line.split(";") for line in lines]
    for i, line in enumerate(lines):
        if isinstance(line, list):
            lines[i] = line[0]
            if len(line) > 1:
                lines.extend(line[1:])

    # Get the tags and their values.
    lines = [line.strip().split("=") for line in lines]
    tags, values = zip(*lines)
    tags = [tag.strip().lower() for tag in tags]
    values = [" ".join(value.strip().split()) for value in values]

    # Put together dictionary and try to evaluate literals.
    incar_tags = {}
    for tag, value in zip(tags, values):
        try:
            incar_tags[tag] = ast.literal_eval(value)
        except ValueError:
            incar_tags[tag] = value

    return incar_tags


def read_poscar(path: os.PathLike) -> Structure:
    """Reads a POSCAR file.

    Parameters
    ----------
    path : str or Path
        Path to the POSCAR file to read in.

    Returns
    -------
    structure : Structure
        A structure object containing the information read from the
        POSCAR.

    See Also
    --------
    write_poscar : Write a POSCAR file.

    Notes
    -----
    The POSCAR file format is described in the `VASP wiki
    <https://www.vasp.at/wiki/index.php/POSCAR>`_.

    Any comments in the POSCAR file are saved as structure attributes.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Save the comment in the structure attributes.
    attr = {"comment": lines[0].strip(), "path": os.path.abspath(path)}

    # Get the scaling factor and apply it to the lattice vectors.
    scaling = float(lines[1])
    _cell = np.zeros((3, 3))
    for i in range(3):
        _cell[i] = list(map(float, lines[2 + i].split()))
    cell = scaling * _cell

    # Get the atomic symbols and their multiplicity/count.
    _kinds = lines[5].split()
    counts = list(map(int, lines[6].split()))
    kinds = np.array([], dtype=str)
    for kind, count in zip(_kinds, counts):
        kinds = np.concatenate((kinds, [kind] * count))

    # Get the atomic positions.
    cartesian = lines[7].startswith(("C", "c"))
    positions = np.zeros((sum(counts), 3))
    for i in range(sum(counts)):
        positions[i] = list(map(float, lines[8 + i].split()))

    structure = Structure(
        kinds=kinds, positions=positions, cell=cell, cartesian=cartesian, attr=attr
    )
    return structure


def read_chg(path: os.PathLike) -> tuple[Structure, np.ndarray]:
    """Reads structure and charge density data from CHG files.

    Parameters
    ----------
    path : str or Path
        Path to a CHG file.

    Returns
    -------
    structure : Structure
        The `Structure` object described in the CHG file.

    data : ndarray
        The charge density data.

    Notes
    -----
    The CHG file format is described in the `VASP wiki
    <https://www.vasp.at/wiki/index.php/CHGCAR>`_.

    """
    # POSCAR like structure header.
    with open(path, "r") as f:
        lines = f.readlines()

    attr = {"comment": lines[0].strip(), "path": os.path.abspath(path)}

    scaling = float(lines[1])
    _cell = np.zeros((3, 3))
    for i in range(3):
        _cell[i] = list(map(float, lines[2 + i].split()))
    cell = scaling * _cell

    _kinds = lines[5].split()
    counts = list(map(int, lines[6].split()))
    kinds = np.array([], dtype=str)
    for kind, count in zip(_kinds, counts):
        np.concatenate((kinds, [kind] * count))

    cartesian = lines[7].startswith(("C", "c"))

    positions = np.zeros((sum(counts), 3))
    for i in range(sum(counts)):
        positions[i] = list(map(float, lines[8 + i].split()))

    structure = Structure(
        kinds=kinds, positions=positions, cell=cell, cartesian=cartesian, attr=attr
    )

    # Data grid shape and charge density data.
    shape = tuple(map(int, lines[9 + sum(counts)].strip().split()))

    _data = " ".join(lines[10 + sum(counts) :]).replace("\n", "")
    data = np.array(list(map(float, _data.split()))).reshape(shape)

    return structure, data


def read_vasprun_xml(path: os.PathLike) -> dict:
    """Reads a vasprun.xml file into a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to a vasprun.xml file.

    Returns
    -------
    vasprun : dict
        A dictionary containing the vasprun.xml data.

    Notes
    -----
    This uses the :meth:`xmltodict.parse` method to parse the XML file.
    The resulting dictionary is not very user friendly (highly nested),
    but it is a good starting point for further processing.

    """
    with open(path, "r") as f:
        vasprun_xml = f.read()

    vasprun_ordered = xmltodict.parse(vasprun_xml, xml_attribs=False)
    vasprun = json.loads(json.dumps(vasprun_ordered))

    return vasprun


# --- Output -----------------------------------------------------------


def write_incar(path: os.PathLike, **incar_tags: dict) -> None:
    """Writes an INCAR file at the given path.

    Parameters
    ----------
    path : str or Path
        Path where to write the INCAR. Any filename will be replaced by
        "INCAR".
    **incar_tags : dict
        INCAR tags and their values.

    See Also
    --------
    read_incar : Read an INCAR file.

    Notes
    -----
    The INCAR tags are not validated. The INCAR is written in the same
    order as the tags are specified. Multiline strings and lists of
    values are handled.

    The comment line contains some provenance information.

    """
    lines = [f"INCAR written by ntcad v{__version__} | {datetime.now()}\n"]
    for tag, value in incar_tags.items():
        line = f"{tag.upper()} = "
        if isinstance(value, (list, tuple)):
            line += " ".join(list(map(str, value)))
        elif isinstance(value, str) and "\n" in value:
            # Multiline strings need enclosing quotes.
            line += f'"{value}"'
        else:
            line += str(value)
        lines.append(line + "\n")

    if os.path.isfile(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "INCAR"), "w") as incar:
        incar.writelines(lines)


def write_poscar(path: os.PathLike, structure: Structure) -> None:
    """Writes a given structure in POSCAR format.

    Parameters
    ----------
    path : str or Path
        Path where to write the POSCAR. If any filename is present, the
        filename is replaced with POSCAR.
    structure
        The `Structure` to write to file.

    Notes
    -----
    The comment line is replaced by a note that the file was written by
    `ntcad` together with the date and time of writing.

    The structure is written in Cartesian coordinates. There is
    currently no option to write in direct coordinates.

    """
    lines = [
        f"POSCAR written by ntcad v{__version__} | {datetime.now()}\n",
        f"{1.0:.5f}\n",
    ]
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

    if os.path.isfile(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "POSCAR"), "w") as poscar:
        poscar.writelines(lines)


def write_potcar(
    path: os.PathLike,
    structure: Structure,
    potentials: dict = None,
    recommended_potentials: bool = False,
) -> None:
    """
    Writes the pseudopotentials needed to simulate the structure.

    Parameters
    ----------
    path : str or Path
        Path where to write the POTCAR. If any filename is present, the
        filename is replaced with POTCAR.
    structure : Structure
        The class:`Structure` instance from which the atomic kinds are
        determined.
    potentials : dict
        A dictionary containing custom potentials that should be used
        instead of the recommended ones. The keys are the atomic kinds
        and the values are the appendices of the POTCAR file names.
    recommended_potentials : bool
        Whether to use the recommended potentials. If `True`, the
        recommended potentials are used. If `False`, the custom
        potentials are used. By default, only the minimal set of
        pseudopotentials is used.

    Notes
    -----
    The available and recommended potentials can be found on the `VASP
    wiki <https://www.vasp.at/wiki/index.php/Available_PAW_potentials>`_.

    The environment variable ``VASP_PP_PATH`` is used to determine the
    location of the pseudopotential files. If the variable is not set,
    an error is raised.

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

    if os.path.isfile(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "POTCAR"), "w") as out_potcar:
        for symbol in np.unique(structure.kinds):
            potcar_folder = symbol + potentials.get(symbol, "")
            potcar_path = os.path.join(base_path, potcar_folder, "POTCAR")
            with open(potcar_path) as in_potcar:
                out_potcar.write(in_potcar.read())


def write_kpoints(
    path: os.PathLike, kpoints: np.ndarray, shift: np.ndarray = None
) -> None:
    """Writes a KPOINTS file.

    Parameters
    ----------
    path : str or Path
        Path at which to write the KPOINTS file. If any filename is present, the
        filename is replaced with KPOINTS.
    kpoints : ndarray
        Either the dimensions of a Monkhorst-Pack grid or a list of
        k-points to be written to file.
    shift : ndarray
        An optional amount to shift the kpoints by. Only applies for
        Monkhorst-Pack grids.

    See Also
    --------
    ntcad.kpoints : Functions for the creation of k-point grids/paths.

    Notes
    -----
    The comment line is replaced by a note that the file was written by
    `ntcad` together with the date and time of writing.

    """
    if not isinstance(kpoints, np.ndarray):
        kpoints = np.array(kpoints)

    lines = [f"KPOINTS written by ntcad v{__version__} | {datetime.now()}\n"]
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
            if len(kpoint) == 3:
                # Automatically append ones if no weights are included.
                kpoint = np.append(kpoint, 1.0)
            lines.append("{:.16f} {:22.16f} {:22.16f} {:22.16f}".format(*kpoint) + "\n")
    else:
        raise ValueError(
            "The kpoints should either describe a Monkhorst-Pack grid "
            "or should contain a list of points in reciprocal space."
        )

    if os.path.isfile(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "KPOINTS"), "w") as kpoints:
        kpoints.writelines(lines)
