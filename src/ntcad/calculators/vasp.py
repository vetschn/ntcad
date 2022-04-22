"""
This module implements a VASP calculator class that can be used as an
interface to write VASP inputs, run VASP jobs and read the resulting
output files.

"""

import os

import numpy as np
from ntcad.calculators.calculator import Calculator
from ntcad.core.structure import Structure


class VASP(Calculator):
    def __init__(
        self,
        directory: os.PathLike,
        structure: Structure,
        kpoints: np.ndarray,
        **kwargs: dict,
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
        """
        super().__init__(directory, structure, **kwargs)

    def calculate(self, command: str) -> None:
        """_summary_

        Parameters
        ----------
        command
            _description_

        Returns
        -------
            _description_
        """
        return super().calculate(command)

    def _write_input(self) -> None:
        """_summary_"""
        pass

    def _read_output(self) -> None:
        """_summary_"""
        pass
