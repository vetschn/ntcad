"""
This module implements a VASP calculator class that can be used as an
interface to write VASP inputs, run VASP jobs and read the resulting
output files.

"""

import os
import subprocess

import numpy as np
from ntcad import Calculator, Structure, vasp


class VASP(Calculator):
    """Calculator interface for running VASP jobs.

    Attributes
    ----------
    directory
        _description_
    incar_tags
        _description_
    structure
        _description_
    kpoints
        _description_
    shift, optional
        _description_, by default None
    potentials, optional
        _description_, by default None
    recommended_potentials, optional
        _description_, by default False

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
        """_summary_

        Parameters
        ----------
        directory
            _description_
        structure
            _description_
        kpoints
            _description_
        shift, optional
            _description_, by default None
        potentials, optional
            _description_, by default None
        recommended_potentials, optional
            _description_, by default False

        """
        self.directory = directory
        self.structure = structure
        self.kpoints = kpoints
        self.shift = shift
        self.potentials = potentials
        self.recommended_potentials = recommended_potentials
        self.incar_tags = incar_tags

    def calculate(self, command: str) -> int:
        """_summary_

        Parameters
        ----------
        command
            _description_

        Returns
        -------
        retcode
            _description_

        """
        self.write_input()

        with open(os.path.join(self.directory, "vasp.out"), "a") as vasp_out:
            retcode = subprocess.call(
                command, shell=True, stdout=vasp_out, cwd=self.directory
            )
        return retcode

    def write_input(self, force: bool = False) -> None:
        """Writes all the inputs file necessary for a VASP run.

        Parameters
        ----------
        force
            If force is set to True, the inputs are overwritten even if
            they are already present.

        """
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # Check if input is written.
        paths = [
            os.path.join(self.directory, file)
            for file in ("INCAR" "POSCAR" "KPOINTS" "POTCAR")
        ]
        if not force and all(os.path.exists(path) for path in paths):
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
        """Clears all outputs of a VASP run."""
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

        pass
