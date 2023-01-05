"""
This module implements a VASP calculator class that can be used as an
interface to write VASP inputs and run VASP jobs.

"""

import os
import subprocess

import numpy as np

from ntcad import vasp
from ntcad.calculator import Calculator
from ntcad.structure import Structure


class VASP(Calculator):
    """
    A calculator class that can be used as an interface to write VASP
    inputs and run VASP jobs.

    Attributes
    ----------
    directory : os.PathLike
        The directory where the VASP run is invoked.
    incar_tags : dict
        A dictionary of tags to be written in the INCAR file.
    structure : Structure
        The structure to be used in the VASP calculation.
    kpoints : np.ndarray
        The kpoints to be used in the VASP calculation.
    shift : dict
        The shift to be used in the KPOINTS file.
    potentials : dict
        A dictionary of potentials to be used in the POTCAR file.
    recommended_potentials : bool
        If True, the recommended potentials are used in the POTCAR file.

    """

    def __init__(
        self,
        directory: os.PathLike,
        structure: Structure,
        kpoints: np.ndarray,
        shift: dict = None,
        potentials: dict = None,
        recommended_potentials: bool = False,
        **incar_tags: dict,
    ) -> None:
        """
        Initializes a VASP calculator.

        Parameters
        ----------
        directory
            The directory where the VASP run is to be invoked.
        structure
            The structure to be used in the VASP calculation.
        kpoints
            The k-points to be used in the VASP calculation. This can
            either be the size of a Monkhorst-Pack grid (e.g. `[21, 21,
            1]`) or a list of k-points in reciprocal space.
        shift
            The shift to be used in the KPOINTS file.
        potentials
            Which potentials to use in the POTCAR file. This is a
            dictionary of the form {element: potential} where
            potential is the name of the potential file.
        recommended_potentials
            If True, the recommended potentials are used in the POTCAR
            file. This is ignored if potentials is not None.
        **incar_tags
            A dictionary of tags to be written to the INCAR file.

        See Also
        --------
        ntcad.vasp.io.write_incar : Writes an INCAR file.
        ntcad.vasp.io.write_poscar : Writes a POSCAR file.
        ntcad.vasp.io.write_kpoints : Writes a KPOINTS file.
        ntcad.vasp.io.write_potcar : Writes a POTCAR file.
        ntcad.core.structure.Structure : An atomic structure.

        """
        self.directory = directory
        self.structure = structure
        self.kpoints = kpoints
        self.shift = shift
        self.potentials = potentials
        self.incar_tags = incar_tags
        self.recommended_potentials = recommended_potentials

    def calculate(self, command: str, overwrite_input: bool = True) -> int:
        """
        Runs a VASP calculation.

        Parameters
        ----------
        command
            The command to run VASP. This should be a command like
            `mpirun -np 4 vasp_std`.
        overwrite_input
            If True, the input files are overwritten even if they are
            already present.

        Returns
        -------
        int
            The return code of the VASP run. A return code of `0` means
            that the calculation was successful. If the run fails, one
            will likely encounter a RuntimeError, but this is not
            guaranteed and you're on your own from there.

        Notes
        -----
        This method invokes the `subprocess.call` method with the
        `shell=True` flag set. See the Python documentation for more
        information.

        The stdout and stderr of the VASP run are redirected to the
        `vasp.out` and `vasp.err` files in the directory where the VASP
        run is invoked.

        """
        self.write_input(overwrite=overwrite_input)

        with open(os.path.join(self.directory, "vasp.out"), "a") as vasp_out:
            with open(os.path.join(self.directory, "vasp.err"), "a") as vasp_err:
                retcode = subprocess.call(
                    command,
                    shell=True,
                    stdout=vasp_out,
                    stderr=vasp_err,
                    cwd=self.directory,
                )
        return retcode

    def write_input(self, overwrite: bool = True) -> None:
        """
        Writes all the inputs file necessary for a VASP run.

        Parameters
        ----------
        overwrite
            If overwrite is set to True, the inputs are overwritten even
            if they are already present.

        """
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # Check if input is written.
        paths = [
            os.path.join(self.directory, file)
            for file in ("INCAR" "POSCAR" "KPOINTS" "POTCAR")
        ]
        if not overwrite and all(os.path.exists(path) for path in paths):
            return

        vasp.io.write_incar(path=self.directory, **self.incar_tags)
        vasp.io.write_poscar(path=self.directory, structure=self.structure)
        vasp.io.write_kpoints(
            path=self.directory, kpoints=self.kpoints, shift=self.shift
        )
        vasp.io.write_potcar(
            path=self.directory,
            structure=self.structure,
            potentials=self.potentials,
            recommended_potentials=self.recommended_potentials,
        )

    def clear(self) -> None:
        """
        Clears all outputs of a VASP run.

        """
        if not os.path.isdir(self.directory):
            raise FileNotFoundError(f"{self.directory} is not a directory.")

        paths = [
            os.path.join(self.directory, file)
            for file in (
                "BSEFATBAND",
                "CHG",
                "CHGCAR",
                "CONTCAR",
                "DOSCAR",
                "EIGENVAL",
                "ELFCAR",
                "IBZKPT",
                "LOCPOT",
                "OSZICAR",
                "OUTCAR",
                "PARCHG",
                "PCDAT",
                "PROCAR",
                "PROOUT",
                "REPORT",
                "TMPCAR",
                "vasprun.xml",
                "vaspout.h5",
                "vaspwave.h5",
                "WAVECAR",
                "WAVEDER",
                "XDATCAR",
            )
        ]

        for p in paths:
            if not os.path.exists(p):
                continue
            os.remove(p)
