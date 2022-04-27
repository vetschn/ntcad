"""
This module implements an OMEN calculator class that can be used as an
interface to write OMEN inputs, run OMEN jobs and read the resulting
output files.

TODO: Since the code is under active development the input verification
should be pretty lax.

"""


import os
from ntcad.calculators.calculator import Calculator
from ntcad.core.structure import Structure


class OMEN(Calculator):
    def __init__(self, directory: os.PathLike, **kwargs: dict) -> None:
        pass

    def calculate(self, command: str) -> None:
        pass

    def write_input(self) -> None:
        pass

    def read_output(self) -> None:
        pass
