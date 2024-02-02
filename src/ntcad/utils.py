"""
Utility functions for all kinds of things.

"""

import subprocess
import warnings
from itertools import product, starmap


def get_idle_hosts(prefix: str) -> list[str]:
    """Finds all hosts that are currently idle.

    Invokes the ``cload`` command to find all hosts that are currently
    idle.

    A host is considered idle if all three load averages are
    below 0.5.

    Parameters
    ----------
    prefix : str
        The prefix of the hosts to search for.

    Returns
    -------
    idle_hosts : list[str]
        A list of all hosts that are currently idle.

    Examples
    --------
    Check which nodes are currently idle:

    >>> get_idle_hosts("node") # doctest: +SKIP
    ["node1", "node2", "node3"] # NOTE: This may take a while.

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
    """A n-dimensional version of the built-in `range` function.

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
        `True`, `start` and `step` are ignored.

    Yields
    ------
    element : tuple[int]
        The next element in the range.

    See Also
    --------
    range : The built-in range function.

    Notes
    -----
    This is taken from `this StackOverflow answer
    <https://stackoverflow.com/a/46332920>`_.

    Examples
    --------
    Create a 2D range generator and convert it to a list:

    >>> it = ndrange((2, 3))
    >>> list(it)
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    One can also use the `start` and `step` arguments:

    >>> it = ndrange((-1, -1), (5, 5), (2, 2))
    >>> list(it)
    [(-1, -1), (-1, 1), (-1, 3), (1, -1), (1, 1), (1, 3), (3, -1), (3, 1), (3, 3)]

    The `centered` argument can be used to create a range centered
    around the origin:

    >>> it = ndrange((3, 3), centered=True)
    >>> list(it)
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

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
    center_index : tuple
        The center index of the object.

    Examples
    --------
    Find the center index of an object with shape (3, 5, 7):

    >>> center_index(shape=(3, 5, 7))
    (1, 2, 3)

    """
    is_even = [s % 2 == 0 for s in shape]
    if any(is_even):
        warnings.warn(
            "Even number of elements in one dimension. Center index is rounded down."
        )
    return tuple((s - 1) // 2 for s in shape)


def get_tuples_summing_to(total: int):
    """Generates all tuples of 3 integers summing to a given total.

    Parameters
    ----------
    total : int
        The sum for which to generate the tuples.

    Yields
    ------
    tuple : tuple[int]
        The next tuple summing to `total`.

    Examples
    --------
    Generate all tuples summing to 2:
    >>> list(get_tuples_summing_to(2))
    [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]

    """

    for i in range(total + 1):
        for j in range(total - i + 1):
            yield i, j, total - i - j
