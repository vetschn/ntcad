""" TODO: Docstrings.

"""

import os

import numpy as np
import ntcad


def write_winput(path: os.PathLike, **winput_tags: dict) -> None:
    """_summary_

    Parameters
    ----------
    path
        _description_
    """
    lines = [f"# winput written by ntcad v{ntcad.__version__} \n"]

    for tag, value in winput_tags.items():
        line = tag
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.array(value)
            if array.ndim == 1:
                line += " = " + " ".join(list(map(str, value)))
            elif array.ndim == 2:
                # Transformation matrices and k-points are a little
                # special.
                line += "\n"
                for i in range(array.shape[0]):
                    for j in range(array.shape[1]):
                        line += "{:10.5f}".format(array[i, j])
                    line += "\n"
            else:
                raise ValueError(
                    f"Value can only be a 1D or 2D array. Instead got: {value=}"
                )
        else:
            line += " = " + str(value)
        lines.append(line + "\n")

    if os.path.isfile(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "winput"), "w") as winput:
        winput.writelines(lines)
