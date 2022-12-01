"""
**utils**
=========
Utility functions for all kinds of things.

"""

import warnings
import subprocess
from itertools import product, starmap
import numpy as np


def get_idle_hosts(prefix: str) -> list[str]:
    """Finds all hosts that are currently idle.

    Invokes the ``cload`` command to find all hosts that are currently
    idle.

    Parameters
    ----------
    prefix : str
        The prefix of the hosts to search for.

    Returns
    -------
    list[str]
        A list of all hosts that are currently idle.

    """
    output = subprocess.check_output(["cload"] + [prefix]).decode()

    lines = output.split("\n")[:-1]
    load_averages = [list(map(float, line.split()[-3:])) for line in lines]
    hosts = [line.split()[0] for line in lines]

    idle_hosts = []
    for load, host in zip(load_averages, hosts):
        if all(val < 0.5 for val in load):
            idle_hosts.append(host)

    return idle_hosts


def ndrange(start, stop=None, step=None, centered=False):
    """A n-dimensional version of the built-in ``range`` function.

    Parameters
    ----------
    start : int or tuple[int]
        The start of the range. If a tuple is given, the range is
        n-dimensional.
    stop : int or tuple[int], optional
        The end of the range. If a tuple is given, the range is
        n-dimensional.
    step : int or tuple[int], optional
        The step of the range. If a tuple is given, the range is
        n-dimensional.
    centered : bool, optional
        Whether the range should be centered around the origin. If
        ``True``, ``start`` and ``step`` are ignored.

    Yields
    ------
    tuple[int]
        The next element in the range.

    See Also
    --------
    range : The built-in range function.

    Notes
    -----
    This is taken from `this StackOverflow answer
    <https://stackoverflow.com/a/46332920>`_.

    """

    if stop is None:
        stop = start
        start = (0,) * len(stop)

    if step is None:
        step = (1,) * len(start)

    if centered:
        center = center_index(stop)
        start = tuple(-c for c in center)
        stop = tuple(c + 1 for c in center)

    if not len(start) == len(stop) == len(step):
        raise ValueError("start, stop, and step must be the same length")

    for index in product(*starmap(range, zip(start, stop, step))):
        yield index


def center_index(shape: tuple) -> tuple:
    """Finds the center index of an array-like object, given its shape.

    Parameters
    ----------
    shape : tuple
        The shape of the object in question.

    Returns
    -------
    tuple
        The center index of the object.

    """
    is_even = [s % 2 == 0 for s in shape]
    if any(is_even):
        warnings.warn(
            "Even number of elements in one dimension. " "Center index is rounded down."
        )
    return tuple((s - 1) // 2 for s in shape)
