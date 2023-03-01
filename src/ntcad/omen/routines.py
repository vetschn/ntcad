"""
This module contains common processing routines and operations for OMEN
calculations.

"""

import multiprocessing

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


def _matrix_info(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Calculates information about the layer/total matrices.

    Parameters
    ----------
    matrix : ndarray
        The layer/total matrix.

    Returns
    -------
    pos : ndarray
        The positions of the atoms.
    kind_inds : ndarray
        The kind indices of the atoms.
    nn : ndarray
        The nearest neighbor matrix.
    num_atoms : int
        The number of atoms.
    max_nn : int
        The maximum number of nearest neighbors.

    See Also
    --------
    max_nn : Calculates the maximum number of nearest neighbors.
    ntcad.omen.io.read_dat : Reads the layer/total matrix.

    """
    pos = matrix[:, :3]
    kind_inds = matrix[:, 3].astype(int) - 1  # MATLAB indexing.
    nn = matrix[:, 4:].astype(int) - 1  # MATLAB indexing.

    num_atoms = pos.shape[0]
    num_nn = nn.shape[-1]

    return pos, kind_inds, nn, num_atoms, num_nn


def _sum_num_orbs(num_orbs: list, matrix: np.ndarray) -> list:
    """Array containing the summed number of orbitals at any atom index.

    Parameters
    ----------
    num_orbs : list
        The number of orbitals per atom kind.
    matrix : ndarray
        The layer/total matrix.

    Returns
    -------
    sum_num_orbs : list
        The summed number of orbitals at any atom index.

    See Also
    --------
    ntcad.omen.io.read_dat : Reads the layer/total matrix.

    """
    num_atoms = matrix.shape[0]
    kind_inds = matrix[:, 3].astype(int) - 1  # MATLAB indexing.

    sum_num_orbs = np.zeros(num_atoms, dtype=int)
    for i in range(num_atoms):
        sum_num_orbs[i] = np.sum(num_orbs[kind_inds[:i]])

    return sum_num_orbs


def split_H_matrices(
    ph_mat_par: dict, H: dict, layer_matrix: np.ndarray, Lz: float
) -> dict:
    """Splits the Hamiltonian matrices into layers given a layer width.

    Parameters
    ----------
    ph_mat_par : dict
        The material parameters.
    H : dict
        The Hamiltonian matrices for every layer.
    layer_matrix : ndarray
        The layer matrix.
    Lz
        The layer width at which to split the Hamiltonian matrices.

    Returns
    -------
    H : dict
        The Hamiltonian matrices for every layer.

    See Also
    --------
    ntcad.omen.io.read_mat_par : Reads the material parameters.
    ntcad.omen.io.read_H_matrices : Reads the Hamiltonian matrices.
    ntcad.omen.io.read_dat : Reads the layer/total matrix.

    """
    layer_pos, kind_inds, *__ = _matrix_info(layer_matrix)
    num_orbs = ph_mat_par["orbitals"]
    sum_num_orbs = _sum_num_orbs(num_orbs, layer_matrix)

    Lz_split = Lz / 2

    slab_a = np.argwhere(layer_pos[:, 2] < Lz_split - 0.01).flatten()
    slab_b = np.argwhere(layer_pos[:, 2] >= Lz_split - 0.01).flatten()

    split_size_a = np.sum(num_orbs[kind_inds[slab_a]])
    split_size_b = np.sum(num_orbs[kind_inds[slab_b]])
    assert split_size_a == split_size_b, "Sanity check"

    # R_a and R_b are block-wise identity matrices.
    R_a = np.zeros((split_size_a, 2 * split_size_a), dtype=int)
    R_b = np.zeros((split_size_a, 2 * split_size_a), dtype=int)

    # NOTE: There may very well be a smarter / more concise way of doing
    # this.
    i = 0
    for i_a, i_b in zip(slab_a, slab_b):
        num_orbs_i = num_orbs[kind_inds[i_a]]  # Number of orbitals on atom i.

        j_a = sum_num_orbs[i_a]
        j_b = sum_num_orbs[i_b]

        R_a[i : i + num_orbs_i, j_a : j_a + num_orbs_i] = np.eye(num_orbs_i)
        R_b[i : i + num_orbs_i, j_b : j_b + num_orbs_i] = np.eye(num_orbs_i)

        i += num_orbs_i

    # Cut the matrices using scipy.sparse for speed.
    R_a = csr_matrix(R_a)
    R_b = csr_matrix(R_b)

    H_split = {}

    H_split[4] = R_a * H[4] * R_a.transpose()

    H_split[5] = R_a * H[4] * R_b.transpose()
    H_split[3] = H_split[5].transpose()

    H_split[6] = R_a * H[5] * R_a.transpose()
    H_split[2] = H_split[6].transpose()

    H_split[7] = R_a * H[5] * R_b.transpose()
    H_split[1] = H_split[7].transpose()

    return H_split


def photon_scattering_matrix_large(
    ph_mat_par: dict,
    M: dict,
    s_layer_matrix: np.ndarray,
    s_total_matrix: np.ndarray,
    s_Lz: float,
    l_layer_matrix: np.ndarray,
    l_total_matrix: np.ndarray,
    l_Lz: float,
    cutoff: float = 2.75,
) -> np.ndarray:
    """Constructs the photon scattering matrix for large systems.

    Parameters
    ----------
    ph_mat_par : dict
        The material parameters.
    M : dict
        The scattering matrices for every direction and layer.
    s_layer_matrix : ndarray
        The small layer matrix.
    s_total_matrix : ndarray
        The small total matrix.
    s_Lz : float
        The small layer width.
    l_layer_matrix : ndarray
        The large layer matrix.
    l_total_matrix : ndarray
        The large total matrix.
    l_Lz : float
        The large layer width.
    cutoff : float, optional
        The cutoff distance for the scattering matrix.

    Returns
    -------
    P : ndarray
        The photon scattering matrix.

    See Also
    --------
    ntcad.omen.io.read_mat_par : Reads the material parameters.
    ntcad.omen.io.read_M_matrices : Reads the scattering matrices.
    ntcad.omen.io.read_dat : Reads the layer/total matrix.

    Notes
    -----
    This function is a direct translation of the MATLAB code. It is
    likely that there are more efficient ways of doing this.

    The ``multiprocessing`` module is used to speed up the calculation
    of the scattering matrix.

    """
    if not (
        s_layer_matrix.shape[-1] == s_total_matrix.shape[-1]
        and l_layer_matrix.shape[-1] == l_total_matrix.shape[-1]
    ):
        raise ValueError("Matrices don't contain the same number of nearest neighbors.")

    s_layer_pos, s_kind_inds, s_layer_nn, num_atoms, num_nn = _matrix_info(
        s_layer_matrix
    )
    l_layer_pos, *__ = _matrix_info(l_layer_matrix)
    s_total_pos, __, s_total_nn, *__ = _matrix_info(s_total_matrix)
    l_total_pos, __, *__ = _matrix_info(l_total_matrix)

    num_orbs = ph_mat_par["orbitals"]
    # s_sum_num_orbs = _sum_num_orbs(num_orbs, s_layer_matrix)
    l_sum_num_orbs = _sum_num_orbs(num_orbs, l_layer_matrix)

    P = np.zeros((num_atoms, 1 + num_nn, 3, max(num_orbs), max(num_orbs)))

    global _compute_P_i

    def _compute_P_i(i: int) -> np.ndarray:
        """Computes the contribution of atom i on ``P``."""
        P_i = np.zeros((1 + num_nn, 3, max(num_orbs), max(num_orbs)))

        num_orbs_i = num_orbs[s_kind_inds[i]]  # Number of orbitals on atom i.

        # Determine the index of atom i in other matrices.
        i_s_total = np.argwhere(
            np.all(np.isclose(s_layer_pos[i], s_total_pos), axis=1)
        ).item()
        i_l_layer = np.argwhere(
            np.all(np.isclose(s_layer_pos[i], l_layer_pos), axis=1)
        ).item()

        for nn_order in range(1 + num_nn):
            if nn_order == 0:
                j = i
                j_s_total = i_s_total
            else:
                j = s_layer_nn[i, nn_order - 1]
                j_s_total = s_total_nn[i_s_total, nn_order - 1]

            Mx = np.zeros((num_orbs_i, max(num_orbs)))
            My = np.zeros((num_orbs_i, max(num_orbs)))
            Mz = np.zeros((num_orbs_i, max(num_orbs)))

            if j == -1:  # No more nearest neighbors. Go to next atom i.
                break

            z = s_total_pos[j_s_total, 2] + 1e-10
            if np.round((z - z % s_Lz) / s_Lz) or (
                np.abs(s_layer_matrix[j, 0] - s_layer_matrix[i, 0]) >= cutoff
            ):
                continue

            num_orbs_j = num_orbs[s_kind_inds[j]]
            coord = s_layer_matrix[j, :2]
            coord_match = np.nonzero(
                np.sqrt(np.sum(np.abs(l_total_matrix[:, :2] - coord) ** 2, axis=1))
                < 1e-3
            )

            for c in np.squeeze(coord_match):
                z = l_total_matrix[c, 2] + 1e-10

                if np.abs((z - s_layer_matrix[j, 2]) % s_Lz) >= 1e-3:
                    continue

                num_ij = int((z - z % l_Lz) / l_Lz) + 4

                l_layer_matrix_coord = l_total_pos[c] - (0, 0, (num_ij - 4) * l_Lz)
                ind_l_layer_matrix = np.argwhere(
                    np.all(np.isclose(l_layer_matrix_coord, l_layer_pos), axis=1)
                ).item()

                slice_i = slice(
                    l_sum_num_orbs[i_l_layer], l_sum_num_orbs[i_l_layer] + num_orbs_i
                )
                slice_j = slice(
                    l_sum_num_orbs[ind_l_layer_matrix],
                    l_sum_num_orbs[ind_l_layer_matrix] + num_orbs_j,
                )

                Mx[:, :num_orbs_j] += np.imag(M["x"][num_ij][slice_i, slice_j])
                My[:, :num_orbs_j] += np.imag(M["y"][num_ij][slice_i, slice_j])
                Mz[:, :num_orbs_j] += np.imag(M["z"][num_ij][slice_i, slice_j])

            P_i[nn_order, 0, :num_orbs_i, :] = Mx
            P_i[nn_order, 1, :num_orbs_i, :] = My
            P_i[nn_order, 2, :num_orbs_i, :] = Mz

        return P_i

    pool = multiprocessing.Pool()
    _P = list(tqdm(pool.imap(_compute_P_i, range(num_atoms)), total=num_atoms))

    # Initialize full Electron-photon scattering matrix. P[i, 0] is the
    # on-site contribution. The atom itself never appears in the
    # nearest-neighbors list.
    P = np.zeros((num_atoms, 1 + num_nn, 3, max(num_orbs), max(num_orbs)))
    for i in range(P.shape[0]):
        P[i, ...] = _P[i]

    return P


def photon_scattering_matrix(
    ph_mat_par: dict,
    M: dict,
    layer_matrix: np.ndarray,
    total_matrix: np.ndarray,
    Lz: float,
) -> np.ndarray:
    """Computes the photon scattering matrix.

    Parameters
    ----------
    ph_mat_par : dict
        Material parameters.
    M : dict
        Electron-photon coupling matrix for every direction and layer.
    layer_matrix : ndarray
        Matrix with the positions of the atoms in the layer.
    total_matrix: ndarray
        Matrix with the positions of the atoms in the total structure.
    Lz : float
        Thickness of the layer.

    Returns
    -------
    P : np.ndarray
        Photon scattering matrix.

    See Also
    --------
    ntcad.omen.io.read_mat_par : Reads the material parameters.
    ntcad.omen.io.read_M_matrices : Reads the scattering matrices.
    ntcad.omen.io.read_dat : Reads the layer/total matrix.

    Notes
    -----
    This function is a direct translation of the Matlab code.

    The ``multiprocessing`` module is used to speed up the computation.

    """
    if not layer_matrix.shape[-1] == total_matrix.shape[-1]:
        raise ValueError("Matrices don't contain the same number of nearest neighbors.")

    layer_pos, kind_inds, layer_nn, num_atoms, num_nn = _matrix_info(layer_matrix)
    total_pos, __, total_nn, *__ = _matrix_info(total_matrix)

    num_orbs = ph_mat_par["orbitals"]
    sum_num_orbs = _sum_num_orbs(num_orbs, layer_matrix)

    # NOTE: The multiprocessing module requires a picklable object in
    # the call to Pool.map. Only functions defined at the module level
    # are picklable, hence the global keyword here.
    # https://docs.python.org/3/library/pickle.html
    global _compute_P_i

    # Atomic sites are treated in parallel.
    def _compute_P_i(i: int) -> np.ndarray:
        """Computes the contribution of atom i on ``P``."""
        P_i = np.zeros((1 + num_nn, 3, max(num_orbs), max(num_orbs)))
        num_orbs_i = num_orbs[kind_inds[i]]  # Number of orbitals on atom i.

        # Determine the index of atom i in the total_matrix.
        i_total = np.argwhere(np.all(layer_pos[i] == total_pos, axis=1)).item()

        # Local interaction terms.
        num_ii = 4  # H_4.bin corresponds to the middle layer.
        slice_i = slice(sum_num_orbs[i], sum_num_orbs[i] + num_orbs_i)
        M_ = [np.imag(M[k][num_ii][slice_i, slice_i]).toarray() for k in "xyz"]
        P_i[0, :, :num_orbs_i, :num_orbs_i] = M_

        # Save the nearest neighbors and the layer number for later.
        js = np.zeros(num_nn, dtype=int)
        nums_ij = np.zeros(num_nn, dtype=int)

        # ij-interaction terms.
        P_ij = np.zeros((num_nn, 3, num_orbs_i, max(num_orbs)))

        # Iterate over all nearest neighbor atoms j of atom i and gather
        # interaction terms.
        for nn_order, j in enumerate(layer_nn[i]):
            if j == -1:  # No more nearest neighbors. Go to next atom i.
                break

            num_orbs_j = num_orbs[kind_inds[j]]  # Number of orbitals on atom j.

            # Determine layer number via z-coordinate.
            z = total_pos[total_nn[i_total, nn_order], 2]
            num_ij = num_ii + int((z - z % Lz) / Lz)

            js[nn_order] = j
            nums_ij[nn_order] = num_ij
            slice_j = slice(sum_num_orbs[j], sum_num_orbs[j] + num_orbs_j)

            M_ = [np.imag(M[k][num_ij][slice_i, slice_j]).toarray() for k in "xyz"]
            P_ij[nn_order, :, :num_orbs_i, :num_orbs_j] = M_

        # Iterate again.
        for nn_order in range(num_nn):
            Mx = np.zeros((num_orbs_i, max(num_orbs)))
            My = np.zeros((num_orbs_i, max(num_orbs)))
            Mz = np.zeros((num_orbs_i, max(num_orbs)))

            if nums_ij[nn_order] == 4:
                sim_inds = np.argwhere(js == js[nn_order])

                for sim_ind in sim_inds:
                    Mx += np.squeeze(P_ij[sim_ind, 0])
                    My += np.squeeze(P_ij[sim_ind, 1])
                    Mz += np.squeeze(P_ij[sim_ind, 2])

            P_i[nn_order + 1, :, :num_orbs_i, :] = [Mx, My, Mz]

        return P_i

    pool = multiprocessing.Pool()
    _P = list(tqdm(pool.imap(_compute_P_i, range(num_atoms)), total=num_atoms))

    # Initialize full Electron-photon scattering matrix. P[i, 0] is the
    # on-site contribution. The atom itself never appears in the
    # nearest-neighbors list.
    P = np.zeros((num_atoms, 1 + num_nn, 3, max(num_orbs), max(num_orbs)))
    for i in range(P.shape[0]):
        P[i, ...] = _P[i]

    return P


def max_nn(nn: np.ndarray) -> int:
    """Returns the maximum number of nearest neighbors.

    Parameters
    ----------
    layer_matrix : ndarray
        Matrix with the positions of the atoms in the layer.

    Returns
    -------
    max_nn : int
        Maximum number of nearest neighbors.

    """
    if not nn.ndim == 2:
        raise ValueError(f"Inconsistent array dimension: {nn.ndim=}")

    nonzeros = np.count_nonzero(nn, axis=1)
    return np.max(nonzeros)
