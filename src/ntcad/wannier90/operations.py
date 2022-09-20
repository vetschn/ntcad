"""
Useful operations to perform on Wannier90 outputs.

"""

import multiprocessing
from typing import Any

import numpy as np
from ntcad.core.kpoints import monkhorst_pack
from scipy import constants
from tqdm import tqdm

c, *__ = constants.physical_constants["speed of light in vacuum"]
e, *__ = constants.physical_constants["elementary charge"]
hbar, *__ = constants.physical_constants["reduced Planck constant in eV s"]
m_e, *__ = constants.physical_constants["electron mass"]


def _midpoint(shape: tuple) -> np.ndarray:
    """Finds the midpoint of the Wigner-Seitz cell indices.

    Parameters
    ----------
    shape
        Shape of the Wigner-Seitz cell indices (``N_1`` x ``N_2`` x
        ``N_3``).

    Returns
    -------
        The Wigner-Seitz cell index of the middle cell.

    """
    return np.floor_divide(np.subtract(shape, 1), 2)


def approximate_position_operator(
    Ai: np.ndarray, centers: np.ndarray, Ra: np.ndarray
) -> np.ndarray:
    """Approximates the position operator elements ``r_R``.

    Note
    ----
    ``N_i`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``A_i``

    Parameters
    ----------
    Ai
        Real-Space lattice vectors (3 x 3).
    centers
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    Ra
        Allowed Wigner-Seitz cell indices.

    Returns
    -------
        The approximated position operator elements (``N_1`` x ``N_2`` x
        ``N_3`` x ``num_wann`` x ``num_wann`` x 3). The indices are
        chosen such that (0, 0, 0) actually gets you the center
        Wigner-Seitz cell position matrix.

    """
    Ras = np.subtract(Ra, np.min(Ra, axis=0))
    N_1, N_2, N_3 = np.max(Ras, axis=0) + 1
    num_wann = centers.shape[0]

    r_R = np.zeros((N_1, N_2, N_3, num_wann, num_wann, 3), dtype=np.complex64)
    for i in range(r_R.shape[-1]):
        d_0_i = centers[:, i].reshape(-1, 1) - centers[:, i]
        for R in Ra:
            r_R[(*R,)][..., i] = (R @ Ai)[i] + d_0_i
    return r_R


def _approximate_momentum_operator(
    H_R: np.ndarray,
    Ai: np.ndarray,
    Ra: np.ndarray = None,
    tau_ij: bool = False,
    centers: np.ndarray = None,
    si_units=False,
) -> np.ndarray:
    """Approximates the momentum operator elements ``p_R``.

    The resulting momentum matrix is in [eV/c] per default. If you wish
    to get the matrix in SI units [kg*m/s], set the ``in_si_units``
    keyword accordingly. OMEN requires [eV/c].

    Note
    ----
    ``N_i`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``A_i``

    Parameters
    ----------
    H_R
        Hamiltonian elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann``
        x ``num_wann``).
    Ai
        Real-Space lattice vectors (3 x 3).
    Ra
        Allowed Wigner-Seitz cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij
        Whether to include the contributions between Wannier centers.
    centers
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    si_units
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.

    Returns
    -------
        The approximated momentum operator elements (``N_1`` x ``N_2`` x
        ``N_3`` x ``num_wann`` x ``num_wann`` x 3). The indices are
        chosen such that (0, 0, 0) actually gets you the center
        Wigner-Seitz cell momentum matrix.

    """
    midpoint = _midpoint(H_R.shape[:3])

    p_R = np.zeros(H_R.shape + (3,), dtype=np.complex64)
    # Iterate over all spacial components.
    for i in tqdm(range(p_R.shape[-1])):
        d_0_i = np.zeros(H_R.shape[-2:])
        if tau_ij:
            # Trickery: Wannier center distances within cell from
            # transposed version of the Wannier centers themselves.
            d_0_i = centers[:, i].reshape(-1, 1) - centers[:, i]
        # Construct the position operator for spacial dimension i.
        r_R_i = np.zeros_like(H_R)
        for Rs in np.ndindex(r_R_i.shape[:3]):
            R = Rs - midpoint
            allowed = np.any(H_R[(*R,)])
            if Ra is not None:
                allowed = np.any(np.all(Ra == R, axis=1))
            if not allowed:
                continue
            r_R_i[(*R,)] += (R @ Ai)[i] + d_0_i
        p_R[..., i] = r_R_i * H_R
    # Conversion to SI units [kg*m/s].
    p_R_SI = 1j * 1e-10 * m_e / hbar * p_R
    if si_units:
        return p_R_SI
    # Conversion to [eV/c].
    return c / e * p_R_SI


def momentum_operator(
    H_R: np.ndarray,
    Ai: np.ndarray,
    r_R: np.ndarray = None,
    approximate: bool = False,
    Ra: np.ndarray = None,
    tau_ij: bool = False,
    centers: np.ndarray = None,
    si_units=False,
) -> np.ndarray:
    """Calculates the momentum operator elements ``p_R``.

    The momentum operator ``p_R`` is the commutator between Hamiltonian
    ``H_R`` and position operator ``r_R`` in the same Wannier basis.

    The resulting momentum matrix is in [eV/c] per default. If you wish
    to get the matrix in SI units [kg*m/s], set the ``in_si_units``
    keyword accordingly. OMEN requires [eV/c].

    Notes
    -----
    ``N_i`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``A_i``.

    The momentum operator components for all spacial dimensions are
    calculated in parallel.


    Parameters
    ----------
    H_R
        Hamiltonian elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann``
        x ``num_wann``).
    Ai
        Real-Space lattice vectors (3 x 3).
    r_R
        Position matrix elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann`` x 3). Not needed if the momentum
        operator should merely be approximated.
    approximate
        Whether to approximate the momentum operator. Defaults to
        ``False``.
    Ra
        Allowed Wigner-Seitz Cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij
        Whether to include the contributions between Wannier centers
        when approximating the momentum operator.
    centers
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    si_units
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.


    Returns
    -------
        Momentum matrix elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann`` x 3). The indices are chosen such
        that (0, 0, 0) actually gets you the center Wigner-Seitz cell
        momentum matrix.

    Notes
    -----
    .. [1] C. Klinkert, "The ab initio microscope: on the performance of
           2D materials as future field-effect transistors", Ph.D.
           thesis, ETH Zurich, 2021.

    """
    if tau_ij and centers is None:
        raise ValueError("Wannier centers needed if ``tau_ij`` is ``True``.")
    if approximate:
        p_R_approximate = _approximate_momentum_operator(
            H_R=H_R,
            Ai=Ai,
            Ra=Ra,
            tau_ij=tau_ij,
            centers=centers,
            si_units=si_units,
        )
        return p_R_approximate
    elif r_R is None:
        raise ValueError("Position Matrix elements needed if ``approx`` is ``False``.")

    # Midpoint of the Wigner-Seitz cell indices.
    midpoint = _midpoint(H_R.shape[:3])

    # NOTE: https://docs.python.org/3/library/pickle.html
    global _compute_p_R_i

    # Spacial dimensions are treated in parallel.
    def _compute_p_R_i(i: int) -> np.ndarray:
        """Computes the i-th spacial contribution to ``p_R``."""
        p_R_i = np.zeros(H_R.shape, dtype=np.complex64)
        d_0_i = np.zeros(H_R.shape[-2:])
        if tau_ij:
            # Trickery: Wannier center distances within cell from
            # transposed version of the Wannier centers themselves.
            d_0_i = centers[:, i].reshape(-1, 1) - centers[:, i]
        # Iterate over all R vectors.
        for Rs in tqdm(np.ndindex(p_R_i.shape[:3])):
            R = Rs - midpoint
            allowed = np.any(H_R[(*R,)])
            if Ra is not None:
                allowed = np.any(np.all(Ra == R, axis=1))
            if not allowed:
                continue
            p_R_i[(*R,)] += H_R[(*R,)] * ((R @ Ai)[i] + d_0_i)
            # Iterate over all R' vectors.
            for Rps in np.ndindex(p_R_i.shape[:3]):
                Rp = Rps - midpoint
                out_of_bounds = np.any(np.abs(R - Rp) > 0) or np.any(np.abs(R + Rp) > 0)
                allowed = np.any(H_R[(*Rp,)]) and np.any(H_R[(*(R - Rp),)])
                if Ra is not None:
                    allowed = np.any(np.all(Ra == Rp, axis=1)) and np.any(
                        np.all(Ra == (R - Rp), axis=1)
                    )
                if out_of_bounds or not allowed:
                    continue
                p_R_i[(*R,)] += H_R[(*Rp,)] @ r_R[(*(R - Rp),)][..., i]
                p_R_i[(*R,)] -= r_R[(*Rp,)][..., i] @ H_R[(*(R - Rp),)]
        return p_R_i

    # Compute the spacial dimensions in parallel.
    pool = multiprocessing.Pool(3)
    _p_R = pool.map(_compute_p_R_i, range(3))

    # Put the momentum matrix together.
    p_R = np.zeros(H_R.shape + (3,), dtype=np.complex64)
    for i in range(p_R.shape[-1]):
        p_R[..., i] = _p_R[i]

    # Conversion to SI units [kg*m/s].
    p_R_SI = 1j * 1e-10 * m_e / hbar * p_R
    if si_units:
        return p_R_SI
    # Conversion to [eV/c].
    return c / e * p_R_SI


def distance_matrix(
    Ai: np.ndarray,
    centers: np.ndarray,
    Ra: np.ndarray,
) -> np.ndarray:
    """Computes the Wannier center distance matrix.

    Parameters
    ----------
    Ai
        Real-Space lattice vectors (3 x 3).
    centers
        List of ``num_wann`` Wannier centers (``num_wann`` x 3) in real
        space.
    Ra
        The allowed Wigner-Seitz cells to calculate the distance matrix
        for (``R_1``, ``R_2``, ``R_3``).

    Returns
    -------
        Matrix (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``) containing distances between Wannier centers for
        all the requested Wigner Seitz cells. The indices are chosen
        such that (0, 0, 0) actually gets you the center Wigner-Seitz
        cell distance matrix.

    """
    Ras = np.subtract(Ra, Ra.min(axis=0))
    N_1, N_2, N_3 = Ras.max(axis=0) + 1

    # Trickery: Wannier center distances within cell from transposed
    # version of the Wannier centers themselves.
    d_0 = centers[:, np.newaxis] - centers

    # Midpoint of the Wigner-Seitz cell indices.
    midpoint = _midpoint((N_1, N_2, N_3))
    num_wann = len(centers)
    d_R = np.zeros((N_1, N_2, N_3, num_wann, num_wann))
    for Rs in np.ndindex((N_1, N_2, N_3)):
        R = Rs - midpoint
        if np.any(np.all(Ra == R, axis=1)):
            d_R[(*R,)] = np.linalg.norm(d_0 + (R @ Ai), axis=2)

    return d_R


def k_sample(
    O_R: np.ndarray,
    kpoints: np.ndarray = None,
    grid_size: tuple = None,
    optimize: Any = "optimal",
) -> np.ndarray:
    """Samples the given operator ``O_R`` at given ``kpoints``.

    Parameters
    ----------
    O_R
        Operator elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).
    kpoints
        Reciprocal-space points in fractional coordinates (``N_k`` x 3).
        Used if ``monkhorst_pack`` is ``None``. This is ideal for path
        sampling.
    grid_size
        Monkhorst-Pack grid size (``N_1``, ``N_2``, ``N_3``). Used if
        ``kpoints`` is ``None``.
    optimize
        Whether to optimize the einsum call. See ``np.einsum`` for more
        information.

    Returns
    -------
        Operator at the specified ``kpoints``.

    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")

    midpoint_R = _midpoint(O_R.shape[:3])
    R = list(np.ndindex(O_R.shape[:3])) - midpoint_R
    R_R = np.zeros(O_R.shape[:3] + (3,))
    for Ri in R:
        R_R[(*Ri,)] = Ri

    if kpoints is not None:
        if grid_size is None:
            raise ValueError("Either 'kpoints' or 'monkhorst_pack' must be specified.")

        R_dot_k = np.einsum("ijkr,lr->ijkl", R_R, kpoints, optimize=optimize)
        phase = np.exp(2j * np.pi * R_dot_k)
        O_k = np.einsum("ijkmn,ijkl->lmn", O_R, phase, optimize=optimize)
        return O_k

    grid_size = tuple(grid_size)
    kpoints = monkhorst_pack(grid_size)
    midpoint_k = _midpoint(grid_size)
    R = list(np.ndindex(grid_size)) - midpoint_k
    k_R = np.zeros(grid_size + (3,))
    for Ri, kpoint in zip(R, kpoints):
        k_R[(*Ri,)] = kpoint

    R_dot_k = np.einsum("ijkr,uvwr->ijkuvw", R_R, k_R, optimize=optimize)
    phase = np.exp(2j * np.pi * R_dot_k)
    O_k = np.einsum("ijkmn,ijkuvw->uvwmn", O_R, phase, optimize=optimize)
    return O_k


def is_hermitian(O_R: np.ndarray) -> bool:
    """Checks whether a given operator is Hermitian.

    Parameters
    ----------
    O_R
        Operator elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).
    Ra
        The allowed Wigner-Seitz cells (``R_1``, ``R_2``, ``R_3``).

    Returns
    -------
        Whether the operator is hermitian or not.

    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")

    midpoint = _midpoint(O_R.shape[:3])

    for Rs in np.ndindex(O_R.shape[:3]):
        R = Rs - midpoint
        if not np.all(np.equal(np.conjugate(O_R[(*R,)].T), O_R[(*-R,)])):
            return False

    return True


def make_hermitian(O_R: np.ndarray) -> np.ndarray:
    """Enforces the given operator to be Hermitian.

    Parameters
    ----------
    O_R
        Operator elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).

    Returns
    -------
        The now hermitian operator (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).

    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")

    midpoint = _midpoint(O_R.shape[:3])

    O_R_hermitian = np.zeros_like(O_R)
    for Rs in np.ndindex(O_R.shape[:3]):
        R = Rs - midpoint
        O_R_hermitian[(*R,)] = 0.5 * (np.conjugate(O_R[(*R,)].T) + O_R[(*-R,)])
        O_R_hermitian[(*-R,)] = np.conjugate(O_R_hermitian[(*R,)].T)

    return O_R_hermitian
