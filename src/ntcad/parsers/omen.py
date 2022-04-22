#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" TODO

"""

import numpy as np
from scipy.sparse import csr_matrix


def read_bin(path: str) -> csr_matrix:
    """Parses an OMEN binary sparse matrix file.

    Parameters
    ----------
    path
        Path to the binary sparse matrix.

    Returns
    -------
    csr_matrix
        The matrix stored in the file as `scipy.sparse.csr_matrix`.

    """
    with open(path, "rb") as f:
        bin = np.fromfile(f, dtype=np.double)

    dim, size, one_indexed = tuple(map(int, bin[:3]))
    data = bin[3:].reshape(size, 4)
    row_ind, col_ind, real, imag = data.T

    if one_indexed:
        row_ind, col_ind = row_ind - 1, col_ind - 1

    matrix = csr_matrix(
        (real + 1j * imag, (row_ind, col_ind)),
        shape=(dim, dim),
        dtype=np.complex64,
    )
    return matrix
