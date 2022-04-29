""" TODO: Docstrings.

"""

import logging

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
    in_si_units=False,
) -> np.ndarray:
    """Approximates the momentum operator elements ``p_R``.

    This function just takes the on-site terms into account.

    The resulting momentum matrix is in [eV/c] per default. If you wish to
    get the matrix in SI units [kg*m/s], set the ``in_si_units`` keyword
    accordingly. OMEN requires [eV/c].

    Note
    ----
    ``N_i`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``A_i``

    Parameters
    ----------
    H_R
        Hamiltonian elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).
    Ai
        Real-Space lattice vectors (3 x 3).
    Ra
        Allowed Wigner-Seitz cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij
        Whether to include the contributions between Wannier centers.
    centers
        Wannier centers (``num_wann`` x 3). Needed to include the ``tau_ij``
        contributions.
    in_si_units
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.

    Returns
    -------
    p_R
        The approximated momentum operator elements (``N_1`` x ``N_2`` x
        ``N_3`` x ``num_wann`` x ``num_wann`` x 3). The indices are chosen
        such that (0, 0, 0) actually gets you the center Wigner-Seitz
        cell distance matrix.

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
            if allowed:
                r_R_i[(*R,)] = (R @ Ai)[i] + d_0_i
        p_R[..., i] = r_R_i * H_R
    # Conversion to SI units [kg*m/s].
    p_R_SI = 1j * 1e-10 * m_e / hbar * p_R
    if in_si_units:
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
    in_si_units=False,
) -> np.ndarray:
    """Calculates the momentum operator elements ``p_R``.

    The momentum operator ``p_R`` is the commutator between Hamiltonian
    ``H_R`` and position operator ``r_R`` in the same Wannier basis.

    The resulting momentum matrix is in [eV/c] per default. If you wish to
    get the matrix in SI units [kg*m/s], set the ``in_si_units`` keyword
    accordingly. OMEN requires [eV/c].

    Note
    ----
    ``N_i`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``A_i``

    Parameters
    ----------
    H_R
        Hamiltonian elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).
    Ai
        Real-Space lattice vectors (3 x 3).
    r_R
        Position matrix elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann`` x 3). Not needed if the momentum operator should
        merely be approximated.
    approximate
        Whether to approximate the momentum operator. Defaults to
        ``False``.
    Ra
        Allowed Wigner-Seitz Cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij
        Whether to include the contributions between Wannier centers.
    centers
        Wannier centers (``num_wann`` x 3). Needed to include the ``tau_ij``
        contributions.
    in_si_units
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.


    Returns
    -------
    p_R
        Momentum matrix elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann`` x 3). The indices are chosen such that (0, 0, 0)
        actually gets you the center Wigner-Seitz cell distance matrix.

    Raises
    ------

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
            in_si_units=in_si_units,
        )
        return p_R_approximate
    elif r_R is None:
        raise ValueError("Position Matrix elements needed if ``approx`` is ``False``.")

    # Midpoint of the Wigner-Seitz cell indices.
    midpoint = np.floor_divide(np.subtract(H_R.shape[:3], 1), 2)

    # Iterate over R, Rp (R') and 3 spatial dimensions and
    # populate the momentum operator matrix.
    p_R = np.zeros(H_R.shape + (3,), dtype=np.complex64)
    for Rs in np.ndindex(p_R.shape[:3]):
        R = Rs - midpoint
        for Rps in np.ndindex(p_R.shape[:3]):
            Rp = Rps - midpoint
            in_bounds_lower = (np.abs(R - Rp) <= midpoint).all()
            in_bounds_upper = (np.abs(R + Rp) <= midpoint).all()
            allowed = (
                np.any(H_R[(*R,)]) and np.any(H_R[(*Rp,)]) and np.any(H_R[(*(R - Rp),)])
            )
            if Ra is not None:
                allowed = (
                    np.any(np.all(Ra == R, axis=1))
                    or np.any(np.all(Ra == Rp, axis=1))
                    or np.any(np.all(Ra == (R - Rp), axis=1))
                )
            if allowed and in_bounds_lower and in_bounds_upper:
                # Iterate over all spacial components.
                for i in range(p_R.shape[-1]):
                    d_0_i = np.zeros(H_R.shape[-2:])
                    if tau_ij:
                        # Trickery: Wannier center distances within cell
                        # from transposed version of the Wannier centers
                        # themselves.
                        d_0_i = centers[:, i].reshape(-1, 1) - centers[:, i]
                    r_R_i = r_R[..., i]
                    p_R_i = np.zeros_like(r_R_i)
                    p_R_i[(*R,)] += H_R[(*R,)] * (R @ Ai)[i] + d_0_i
                    p_R_i[(*R,)] += H_R[(*Rp,)] @ r_R_i[(*(R - Rp),)]
                    p_R_i[(*R,)] -= r_R_i[(*Rp,)] @ H_R[(*(R - Rp),)]
                    p_R[..., i] += p_R_i
    # Conversion to SI units [kg*m/s].
    p_R_SI = 1j * 1e-10 * m_e / hbar * p_R
    if in_si_units:
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
        Matrix (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x ``num_wann``)
        containing distances between Wannier centers for all the
        requested Wigner Seitz cells. The indices are chosen such that
        (0, 0, 0) actually gets you the center Wigner-Seitz cell
        distance matrix.

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
