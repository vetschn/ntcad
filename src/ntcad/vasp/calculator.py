"""
This module implements a VASP calculator class that can be used as an
interface to write VASP inputs, run VASP jobs and read the resulting
output files.

"""

import os
import subprocess

import numpy as np
from ntcad.core.calculator import Calculator
from ntcad.core.structure import Structure
from ntcad.vasp.io import (
    read_incar,
    read_poscar,
    write_incar,
    write_kpoints,
    write_poscar,
    write_potcar,
)


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
        self.write_input()

        with open(os.path.join(self.directory, "vasp.out"), "a") as vasp_out:
            # TODO: Don't really like the shell=True here.
            retcode = subprocess.call(
                command, shell=True, stdout=vasp_out, cwd=self.directory
            )
        return retcode

    def write_input(self) -> None:
        """Writes all the inputs file necessary for a VASP run.

        This includes INCAR, POSCAR, KPOINTS, and POTCAR.

        """
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        # Check if input is written.
        paths = [
            os.path.join(self.directory, file)
            for file in ("INCAR" "POSCAR" "KPOINTS" "POTCAR")
        ]
        if all(os.path.exists(path) for path in paths):
            return

        write_incar(path=self.directory, **self.incar_tags)
        write_poscar(path=self.directory, structure=self.structure)
        write_kpoints(path=self.directory, kpoints=self.kpoints, shift=self.shift)
        write_potcar(
            path=self.directory,
            structure=self.structure,
            potentials=self.potentials,
            recommended_potentials=self.recommended_potentials,
        )

    def reset(self) -> None:
        """TODO: Clears all outputs of a VASP run."""
        pass
