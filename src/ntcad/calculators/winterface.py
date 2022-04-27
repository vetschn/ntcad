"""
This module implements a Winterface calculator class that can be used as
an interface to run Winterface jobs.

"""

import os

from ntcad.calculators.calculator import Calculator


class Winterface(Calculator):
    def __init__(self, directory: os.PathLike, **kwargs: dict) -> None:
        super().__init__(directory, **kwargs)

    def calculate(self, command: str) -> None:
        return super().calculate(command)

    def _write_input(self) -> None:
        return super()._write_input()

    def _read_output(self) -> None:
        return super()._read_output()

    pass
