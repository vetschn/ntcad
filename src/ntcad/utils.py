"""
Utility functions for all kinds of things.

"""

import subprocess


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
