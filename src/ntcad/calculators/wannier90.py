"""
This module implements a Wannier90 calculator class that can be used as
an interface to run Wannier90 jobs and read the resulting output files.

"""

import os

from ntcad.calculators.calculator import Calculator
from ntcad.core.structure import Structure


class Wannier90(Calculator):
    """_summary_

    NOTE: This should take the output of a DFT calculation as input,
    probably easiest to just get the directory of the DFT -> Wannier90
    calculation. This calculator does not write input files per se
    (everything in seedname.win can be set via vasp-wannier incar tags.)

    Attributes
    ----------


    Methods
    -------

   """
    def __init__(
        self, directory: os.PathLike, structure: Structure, **kwargs: dict
    ) -> None:
        super().__init__(directory, structure, **kwargs)

    def calculate(self, command: str) -> None:
        return super().calculate(command)

    def _write_input(self) -> None:
        return super()._write_input()

    def _read_output(self) -> None:
        return super()._read_output()
