"""
OMEN Calculator
===============

This module implements an OMEN calculator class that can be used as an
interface to write OMEN inputs, run OMEN jobs and read the resulting
output files.

.. warning::

    Nothing here is implemented yet.

"""

import os

from ntcad import Calculator


class OMEN(Calculator):
    """OMEN calculator class.

    .. warning::

        Nothing here is implemented yet.

    """
    def __init__(self, directory: os.PathLike, **kwargs: dict) -> None:
        pass

    def calculate(self, command: str) -> None:
        pass

    def write_input(self) -> None:
        pass
