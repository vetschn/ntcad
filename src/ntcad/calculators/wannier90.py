"""
This module implements a Wannier90 calculator class that can be used as
an interface to run Wannier90 jobs and read the resulting output files.

Generating Wannier90 input files is left for another time.
"""

from ntcad.core.structure import Structure
from ntcad.calculators.calculator import Calculator
import os

class Wannier90(Calculator):

    def __init__(self, directory: os.PathLike, structure: Structure, **kwargs: dict) -> None:
        super().__init__(directory, structure, **kwargs)

    def calculate(self, command: str) -> None:
        return super().calculate(command)
    
    def _write_input(self) -> None:
        return super()._write_input()

    def _read_output(self) -> None:
        return super()._read_output()
