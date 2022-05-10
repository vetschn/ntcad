"""
This module is intended to implement an abstract template for all
specific calculator implementations.

"""

import os
from abc import ABC, abstractmethod


class Calculator(ABC):
    """Abstract base class for calculator implementations.

    TODO

    - A calculator knows the directory the calculation should be
      performed in.

    - A calculator should implement a ``calculate`` method that starts
      the calculation. This sort of method should take a string that
      defines the exact command to be executed command line.

    - A calculator should take input arguments and write them to the
      corresponding input files. It's assumed that all codes read some
      input files.

    - A calculator should be able to parse all calculation outputs and
      store them as accessible properties. It should be checked if a
      property is None and the corresponding file containing the
      information should only be read in this case. This should make
      things a little bit less slow (not high priority).

    - While ASE is more atoms/structure centered, the aim here is to
      make things more calculator / codes centered. We are more
      interested in the calculated properties of a structure than in the
      structure itself.

    Attributes
    ----------
    dir
        The directory the calculation inputs and outputs should be
        generated in.
    structure
        The

    Methods
    -------
    calculate
        Start the calculation according to the specified command.

    """

    def __init__(self, directory: os.PathLike, **kwargs: dict) -> None:
        """_summary_

        Parameters
        ----------
        directory
            _description_
        structure
            _description_
        """
        ...

    @abstractmethod
    def calculate(self, command: str) -> None:
        """_summary_

        Parameters
        ----------
        command
            _description_
        """
        ...

    @abstractmethod
    def write_input(self) -> None:
        """_summary_"""
        ...
