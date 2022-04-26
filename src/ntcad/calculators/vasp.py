"""
This module implements a VASP calculator class that can be used as an
interface to write VASP inputs, run VASP jobs and read the resulting
output files.

"""

import os
import subprocess

import numpy as np
from ntcad.calculators.calculator import Calculator
from ntcad.core.structure import Structure
from ntcad.io.vasp import write_incar, write_kpoints, write_poscar, write_potcar


class VASP(Calculator):
    """_summary_

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

    Methods
    -------
    calculate
        _description_
    read_output
        _description_
    write_input
        _description_


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
        # Check if input is written.
        paths = [
            os.path.join(self.directory, file)
            for file in ("INCAR" "POSCAR" "KPOINTS" "POTCAR")
        ]
        files_exist = [os.path.exists(path) for path in paths]
        if not all(files_exist):
            self.write_input()

        with open("vasp.out", "a") as vasp_out:
            # TODO: Don't really like the shell=True here.
            retcode = subprocess.call(
                command, shell=True, stdout=vasp_out, cwd=self.directory
            )
        return retcode

    def write_input(self) -> None:
        """Writes all the inputs file necessary for a VASP run.

        This includes INCAR, POSCAR, KPOINTS, and POTCAR.

        """
        write_incar(self.directory, **self.incar_tags)
        write_poscar(self.directory, self.structure)
        write_kpoints(self.directory, self.kpoints, self.shift)
        write_potcar(
            self.directory, self.structure, self.potentials, self.recommended_potentials
        )

    def read_output(self) -> None:
        """_summary_"""
        pass

    def reset(self) -> None:
        """Clears all outputs of a VASP run."""
        pass
