"""
This module abstracts reading from and writing from different file types
for all external calculators by trying to guess file types.

"""
import os
from typing import Any

from ntcad import omen, vasp, wannier90, winterface


def read(path: os.PathLike, filetype: str = None, **kwargs: dict) -> Any:
    """Tries to read any implemented filetype by guessing file format.

    Parameters
    ----------
    path
        Path to the file to be read in.
    filetype, optional
        Specify this keyword to explicitly set the filetype. If None,
        the function tries to guess the filetype.

    Returns
    -------
        The specified file.

    Raises
    ------
    NotImplementedError
        If the filetype could not be guessed, or if the explicitly
        specified filetype is unknown.

    """
    filename = str(os.path.basename(path))
    __, filext = os.path.splitext(filename)

    # --- TODO: OMEN filetypes. ----------------------------------------
    if filext == ".bin" or filetype == "bin":
        return omen.io.read_bin(path)

    if filename == "Layer_Matrix.dat" or filetype == "layer_matrix":
        return omen.io.read_layer_matrix_dat(path)

    if filename == "lattice_dat" or filetype == "lattice_dat":
        return omen.io.read_lattice_dat(path)

    if filename.endswith("mat_dat") or filetype == "mat_par":
        return omen.io.read_mat_par(path)

    # --- VASP filetypes -----------------------------------------------
    if filename == "POSCAR" or filetype == "poscar":
        return vasp.io.read_poscar(path)
        # TODO

    # --- TODO: Wannier90 filetypes ------------------------------------
    if filename.endswith("_hr.dat") or filetype == "hr_dat":
        full = kwargs.get("full")
        return wannier90.io.read_hr_dat(path, full=full)

    if filename.endswith("_r.dat") or filetype == "r_dat":
        full = kwargs.get("full")
        return wannier90.io.read_r_dat(path, full=full)

    if filename.endswith("_band.dat") or filetype == "band_dat":
        return wannier90.io.read_band_dat(path)

    if filename.endswith("_band.kpt") or filetype == "band_kpt":
        return wannier90.io.read_band_kpt(path)

    if filext == ".eig" or filetype == "eig":
        return wannier90.io.read_eig(path)

    if filext == ".wout" or filetype == "wout":
        return wannier90.io.read_wout(path)

    # --- TODO: Winterface filetypes -----------------------------------
    pass

    if filetype is None:
        raise NotImplementedError(
            "Filetype could not be detected automatically. Try explicitly "
            "setting `filetype`."
        )
    raise NotImplementedError(
        f"Filetype could not be detected automatically and `filetype` "
        f"{filetype} is not recognized."
    )
