""" TODO: Docstrings.


NOTE: Beware of numpy advanced indexing pitfalls.

"""

import logging
import multiprocessing

import numpy as np
from scipy import constants

logger = logging.Logger(__name__)

c, *__ = constants.physical_constants["speed of light in vacuum"]
e, *__ = constants.physical_constants["elementary charge"]
hbar, *__ = constants.physical_constants["reduced Planck constant in eV s"]
m_e, *__ = constants.physical_constants["electron mass"]


def _approximate_momentum_operator(
    H_R: np.ndarray,
    Ai: np.ndarray,
    Ra: np.ndarray = None,
    tau_ij: bool = False,
    centers: np.ndarray = None,
    si_units=False,
) -> np.ndarray:
    """Approximates the momentum operator elements ``p_R``.

    This function just takes the on-site terms into account.

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
    Ra, optional
        Allowed Wigner-Seitz cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij, optional
        Whether to include the contributions between Wannier centers.
    centers, optional
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    si_units, optional
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.

    Returns
    -------
    p_R
        The approximated momentum operator elements (``N_1`` x ``N_2`` x
        ``N_3`` x ``num_wann`` x ``num_wann`` x 3). The indices are
        chosen such that (0, 0, 0) actually gets you the center
        Wigner-Seitz cell distance matrix.

    """
    # Midpoint of the Wigner-Seitz cell indices.
    midpoint = np.floor_divide(np.subtract(H_R.shape[:3], 1), 2)

    p_R = np.zeros(H_R.shape + (3,), dtype=np.complex64)
    # Iterate over all spacial components.
    for i in range(p_R.shape[-1]):
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
    r_R, optional
        Position matrix elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann`` x 3). Not needed if the momentum
        operator should merely be approximated.
    approximate, optional
        Whether to approximate the momentum operator. Defaults to
        ``False``.
    Ra, optional
        Allowed Wigner-Seitz Cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij, optional
        Whether to include the contributions between Wannier centers
        when approximating the momentum operator.
    centers, optional
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    si_units, optional
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.


    Returns
    -------
    p_R
        Momentum matrix elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann`` x 3). The indices are chosen such
        that (0, 0, 0) actually gets you the center Wigner-Seitz cell
        distance matrix.

    Raises
    ------
    ValueError
        _description_

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
    midpoint = np.floor_divide(np.subtract(H_R.shape[:3], 1), 2)

    # NOTE: The multiprocessing module requires a picklable object in
    # the call to Pool.map. Only funcions defined at the module level
    # are picklable, hence the global keyword here.
    # https://docs.python.org/3/library/pickle.html
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
        for Rs in np.ndindex(p_R_i.shape[:3]):
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
    d_R
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
    midpoint = np.floor_divide(np.subtract((N_1, N_2, N_3), 1), 2)
    num_wann = len(centers)
    d_R = np.zeros((N_1, N_2, N_3, num_wann, num_wann))
    for Rs in np.ndindex((N_1, N_2, N_3)):
        R = Rs - midpoint
        if np.any(np.all(Ra == R, axis=1)):
            d_R[(*R,)] = np.linalg.norm(d_0 + (R @ Ai), axis=2)

    return d_R


def k_sample(O_R: np.ndarray, kpoints: np.ndarray) -> np.ndarray:
    """Samples the given operator ``O_R`` at given ``kpoints``.

    Parameters
    ----------
    O_R
        Operator elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).
    kpoints
        Reciprocal-space points in fractional coordinates (``N_k`` x 3).

    Returns
    -------
    O_k
        Operator at the specified ``kpoints``.

    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")
    # TODO: This here could definitely be done in a nicer / more concise
    # way.
    midpoint = np.floor_divide(np.subtract(O_R.shape[:3], 1), 2)
    R = list(np.ndindex(O_R.shape[:3])) - midpoint
    R_R = np.zeros(O_R.shape[:3] + (3,))
    for Ri in R:
        R_R[(*Ri,)] = Ri
    phase = np.exp(2j * np.pi * np.einsum("ijkr,lr->ijkl", R_R, kpoints))
    O_k = np.einsum("ijkmn,ijkl->lmn", O_R, phase)
    return O_k


def is_hermitian(O_R: np.ndarray) -> bool:
    """_summary_

    Parameters
    ----------
    O_R
        _description_
    Ra, optional
        _description_, by default None

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")

    midpoint = np.floor_divide(np.subtract(O_R.shape[:3], 1), 2)

    hermitian = True
    for Rs in np.ndindex(O_R.shape[:3]):
        R = Rs - midpoint
        if not np.all(np.equal(np.conjugate(O_R[(*R,)].T), O_R[(*-R,)])):
            hermitian = False

    return hermitian


def make_hermitian(O_R: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    O_R
        _description_

    Returns
    -------
        _description_
    """
    if O_R.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {O_R.ndim=}")

    midpoint = np.floor_divide(np.subtract(O_R.shape[:3], 1), 2)

    O_R_hermitian = np.zeros_like(O_R)
    for Rs in np.ndindex(O_R.shape[:3]):
        R = Rs - midpoint
        O_R_hermitian[(*R,)] = 0.5 * (np.conjugate(O_R[(*R,)].T) + O_R[(*-R,)])
        O_R_hermitian[(*-R,)] = np.conjugate(O_R_hermitian[(*R,)].T)

    return O_R_hermitian
