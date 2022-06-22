"""
This module implements a Winterface calculator class that can be used as
an interface to run Winterface jobs.

"""

import os
import subprocess

from ntcad import Calculator
from ntcad.winterface.io import write_winput


class Winterface(Calculator):
    """_summary_

    Parameters
    ----------
    Calculator
        _description_
    """

    def __init__(self, directory: os.PathLike, **winput_tags: dict) -> None:
        """_summary_

        Parameters
        ----------
        directory
            _description_
        """
        self.directory = directory
        self.winput_tags = winput_tags

    def calculate(self, command: str) -> int:
        """_summary_

        Parameters
        ----------
        command
            _description_

        Returns
        -------
        int
            _description_
        """
        self.write_input()

        # TODO: Don't really like the shell=True here.
        retcode = subprocess.call(command, shell=True, cwd=self.directory)
        return retcode

    def write_input(self) -> None:
        """_summary_"""
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        if os.path.exists(os.path.join(self.directory, "winput")):
            return

        write_winput(path=self.directory, **self.winput_tags)

    def read_output(self) -> None:
        """_summary_

        """
        ...

    pass
