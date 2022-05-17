"""_summary_
"""

import subprocess


def get_idle_hosts(name: str) -> list[str]:
    """_summary_

    Parameters
    ----------
    name
        _description_

    Returns
    -------
        _description_
    """
    output = subprocess.check_output(["cload"] + [name]).decode()
    lines = output.split("\n")[:-1]
    load_averages = [list(map(float, line.split()[-3:])) for line in lines]
    hosts = [line.split()[0] for line in lines]
    idle_hosts = []
    for load, host in zip(load_averages, hosts):
        if all(val < 0.5 for val in load):
            idle_hosts.append(host)
    return idle_hosts
