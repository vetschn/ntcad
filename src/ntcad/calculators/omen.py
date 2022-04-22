"""
This module implements an OMEN calculator class that can be used as an
interface to write OMEN inputs, run OMEN jobs and read the resulting
output files.

TODO: Since the code is under active development the input verification
should be pretty lax.

"""


from ntcad.core.structure import Structure
from ntcad.calculators.calculator import Calculator
import os

class OMEN(Calculator):

    def __init__(self, directory: os.PathLike, structure: Structure, **kwargs: dict) -> None:
        super().__init__(directory, structure, **kwargs)

    def calculate(self, command: str) -> None:
        return super().calculate(command)
    
    def _write_input(self) -> None:
        return super()._write_input()

    def _read_output(self) -> None:
        return super()._read_output()