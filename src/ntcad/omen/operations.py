"""TODO
"""

import numpy as np
from scipy.sparse import csr_matrix


def electron_photon_coupling_matrix(layer_matrix: np.ndarray, ph_mat_par: dict) -> csr_matrix:
    """_summary_

    Parameters
    ----------
    layer_matrix
        _description_
    ph_mat_par
        _description_

    Returns
    -------
        _description_
    """
    num_orbitals = ph_mat_par["num_orbitals"]
    


def max_nearest_neighbors(nearest_neighbors: np.ndarray) -> int:
    """_summary_

    Parameters
    ----------
    layer_matrix
        _description_

    Returns
    -------
        _description_
    """
    if not nearest_neighbors.ndim == 2:
        raise ValueError(f"Inconsistent array dimension: {nearest_neighbors.ndim=}")

    nonzeros = np.count_nonzero(nearest_neighbors, axis=1)
    return np.max(nonzeros)