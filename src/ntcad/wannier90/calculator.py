"""
This module implements a Wannier90 calculator class that can be used as
an interface to run Wannier90 jobs.

"""


import os

from ntcad.calculator import Calculator
from ntcad.structure import Structure


class Wannier90(Calculator):
    """Calculator interface for running Wannier90 jobs.

    NOTE: This should take the output of a DFT calculation as input,
    probably easiest to just get the directory of the DFT -> Wannier90
    calculation. This calculator does not write input files per se
    (everything in seedname.win can be set via vasp-wannier incar tags.)

    The same should be true for other DFT codes (still have to check).

    """

    def __init__(
        self, directory: os.PathLike, structure: Structure, **kwargs: dict
    ) -> None:
        pass

    def calculate(self, command: str) -> None:
        return super().calculate(command)

    def write_input(self) -> None:
        return super()._write_input()

    def read_output(self) -> None:
        return super()._read_output()
