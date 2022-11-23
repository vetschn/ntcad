"""
The :obj:`ntcad.omen` package contains a calculator for the OMEN code
(:mod:`ntcad.omen.calculator`), file I/O routines for OMEN files
(:mod:`ntcad.omen.io`), and useful data processing methods
(:mod:`ntcad.omen.operations`).

.. warning::

    Some of the code/functionality in this package may only be
    half-baked.

"""

from ntcad.omen.calculator import OMEN
from ntcad.omen.operations import (
    max_nn,
    photon_scattering_matrix,
    photon_scattering_matrix_large,
    split_H_matrices,
)
