"""TODO
"""

import multiprocessing
import os
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm


def split_H_matrices(
    ph_mat_par: dict, H: dict, layer_matrix: np.ndarray, Lz: float
) -> dict:
    """_summary_

    Parameters
    ----------
    ph_mat_par
        _description_
    H
        _description_
    layer_matrix
        _description_
    Lz
        _description_

    Returns
    -------
        _description_

    """
    # Extracting some useful information.
    num_orbs = ph_mat_par["orbitals"]

    layer_pos = layer_matrix[:, :3]
    kind_inds = layer_matrix[:, 3].astype(int) - 1

    num_atoms = layer_pos.shape[0]

    # Array containing the summed number of orbitals at any atom index.
    sum_num_orbs = np.zeros(num_atoms, dtype=int)
    for i in range(num_atoms):
        sum_num_orbs[i] = np.sum(num_orbs[kind_inds[:i]])

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


def photon_scattering_matrix(
    ph_mat_par: dict,
    M: dict,
    layer_matrix: np.ndarray,
    total_matrix: np.ndarray,
    Lz: float,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    ph_mat_par
        _description_
    M
        _description_
    layer_matrix
        _description_
    total_matrix
        _description_
    Lz
        _description_

    Returns
    -------
        _description_
    """
    # Extracting some useful information.
    num_orbs = ph_mat_par["orbitals"]

    layer_pos = layer_matrix[:, :3]
    kind_inds = layer_matrix[:, 3].astype(int) - 1
    layer_nn = layer_matrix[:, 4:].astype(int) - 1

    total_pos = total_matrix[:, :3]
    total_nn = (total_matrix[:, 4:] - 1).astype(int)

    num_atoms = layer_pos.shape[0]
    assert layer_nn.shape[-1] == total_nn.shape[-1], "Sanity check"
    num_nn = layer_nn.shape[-1]

    # Array containing the summed number of orbitals at any atom index.
    sum_num_orbs = np.zeros(num_atoms, dtype=int)
    for i in range(num_atoms):
        sum_num_orbs[i] = np.sum(num_orbs[kind_inds[:i]])

    # NOTE: The multiprocessing module requires a picklable object in
    # the call to Pool.map. Only functions defined at the module level
    # are picklable, hence the global keyword here.
    # https://docs.python.org/3/library/pickle.html
    global _compute_P_i

    # Spacial dimensions are treated in parallel.
    def _compute_P_i(i: int) -> np.ndarray:
        """Computes the contribution of atom i on ``P``."""
        P_i = np.zeros((1 + num_nn, 3, max(num_orbs), max(num_orbs)))
        num_orbs_i = num_orbs[kind_inds[i]]  # Number of orbitals on atom i.

        # Determine the index of atom i in the total_matrix.
        i_total = np.argwhere(np.all(layer_pos[i] == total_pos, axis=1)).item()

        # Local interaction terms.
        num_ii = 4  # H_4.bin corresponds to the middle layer.
        slice_i = slice(sum_num_orbs[i], sum_num_orbs[i] + num_orbs_i)

        Mx = np.imag(M["x"][num_ii][slice_i, slice_i]).toarray()
        My = np.imag(M["y"][num_ii][slice_i, slice_i]).toarray()
        Mz = np.imag(M["z"][num_ii][slice_i, slice_i]).toarray()

        P_i[0, :, :num_orbs_i, :num_orbs_i] = [Mx, My, Mz]

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

            Mx = np.imag(M["x"][num_ij][slice_i, slice_j]).toarray()
            My = np.imag(M["y"][num_ij][slice_i, slice_j]).toarray()
            Mz = np.imag(M["z"][num_ij][slice_i, slice_j]).toarray()

            P_ij[nn_order, :, :num_orbs_i, :num_orbs_j] = [Mx, My, Mz]

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
