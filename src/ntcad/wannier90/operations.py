"""
Useful operations to perform on Wannier90 outputs.

"""

import multiprocessing
from typing import Any

import numpy as np
from ntcad.core.kpoints import monkhorst_pack
from scipy import constants
from ntcad.utils import ndrange
import scipy.linalg as spla
from tqdm import tqdm

c, *__ = constants.physical_constants["speed of light in vacuum"]
e, *__ = constants.physical_constants["elementary charge"]
hbar, *__ = constants.physical_constants["reduced Planck constant in eV s"]
m_e, *__ = constants.physical_constants["electron mass"]


def approximate_position_operator(
    Ai: np.ndarray, centers: np.ndarray, Ra: np.ndarray
) -> np.ndarray:
    """Approximates the position operator elements ``rR``.

    Parameters
    ----------
    Ai : np.ndarray
        Real-Space lattice vectors (3 x 3).
    centers : np.ndarray
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    Ra : np.ndarray
        Allowed Wigner-Seitz cell indices.

    Returns
    -------
    np.ndarray
        The approximated position operator.

    Notes
    -----
    ``Ni`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``Ai``.

    """
    Ras = np.subtract(Ra, np.min(Ra, axis=0))
    N1, N2, N3 = np.max(Ras, axis=0) + 1
    num_wann = centers.shape[0]

    rR = np.zeros((N1, N2, N3, num_wann, num_wann, 3), dtype=np.complex64)
    for i in range(rR.shape[-1]):
        d_0_i = centers[:, i].reshape(-1, 1) - centers[:, i]
        for R in Ra:
            rR[tuple(R)][..., i] = (R @ Ai)[i] + d_0_i
    return rR


def _approximate_momentum_operator(
    hR: np.ndarray,
    Ai: np.ndarray,
    Ra: np.ndarray = None,
    tau_ij: bool = False,
    centers: np.ndarray = None,
    si_units=False,
) -> np.ndarray:
    """Approximates the momentum operator elements ``pR``.

    The resulting momentum matrix is in [eV/c] per default. If you wish
    to get the matrix in SI units [kg*m/s], set the ``si_units``
    keyword accordingly. OMEN requires [eV/c].

    Parameters
    ----------
    hR : np.ndarray
        Hamiltonian in position basis.
    Ai : np.ndarray
        Real-Space lattice vectors (3 x 3).
    Ra : np.ndarray
        Allowed Wigner-Seitz cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij : bool
        Whether to include the contributions between Wannier centers.
    centers : np.ndarray
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    si_units : bool
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        The approximated momentum operator.

    Notes
    -----
    ``Ni`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``Ai``.

    """
    pR = np.zeros(hR.shape + (3,), dtype=np.complex64)
    for i in range(3):  # x, y, z.
        d0_i = np.zeros(hR.shape[-2:])
        if tau_ij:
            d0_i = centers[:, i].reshape(-1, 1) - centers[:, i]
        # Construct the position operator for spacial dimension i.
        rR_i = np.zeros_like(hR)
        for R in ndrange(rR_i.shape[:3], centered=True):
            allowed = np.any(hR[R])
            if Ra is not None:
                allowed = np.any(np.all(Ra == R, axis=1))
            if not allowed:
                continue
            rR_i[R] += (R @ Ai)[i] + d0_i
        pR[..., i] = rR_i * hR
    # Conversion to SI units [kg*m/s].
    pR_SI = 1j * 1e-10 * m_e / hbar * pR
    if si_units:
        return pR_SI
    # Conversion to [eV/c].
    return c / e * pR_SI


def momentum_operator(
    hR: np.ndarray,
    Ai: np.ndarray,
    rR: np.ndarray = None,
    approximate: bool = False,
    Ra: np.ndarray = None,
    tau_ij: bool = False,
    centers: np.ndarray = None,
    si_units=False,
) -> np.ndarray:
    """Calculates the momentum operator elements ``pR``.

    The momentum operator ``pR`` is the commutator between Hamiltonian
    ``H_R`` and position operator ``rR`` in the same Wannier basis.

    The resulting momentum matrix is in [eV/c] per default. If you wish
    to get the matrix in SI units [kg*m/s], set the ``in_si_units``
    keyword accordingly. OMEN requires [eV/c].

    Parameters
    ----------
    hR : np.ndarray
        Hamiltonian elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann``
        x ``num_wann``).
    Ai : np.ndarray
        Real-Space lattice vectors (3 x 3).
    rR : np.ndarray
        Position matrix elements (``N_1`` x ``N_2`` x ``N_3`` x
        ``num_wann`` x ``num_wann`` x 3). Not needed if the momentum
        operator should merely be approximated.
    approximate : bool
        Whether to approximate the momentum operator. Defaults to
        ``False``.
    Ra : np.ndarray
        Allowed Wigner-Seitz Cell indices. If not given, this assumes
        that all completely zero Hamiltonian blocks are not allowed.
    tau_ij : bool
        Whether to include the contributions between Wannier centers
        when approximating the momentum operator.
    centers : np.ndarray
        Wannier centers (``num_wann`` x 3). Needed to include the
        ``tau_ij`` contributions.
    si_units : bool
        Whether to return the momentum operator in SI units [kg*m/s].
        Defaults to ``False``.


    Returns
    -------
    np.ndarray
        Momentum matrix.

    Notes
    -----
    ``N_i`` correspond to the number of Wigner-Seitz cells along the
    lattice vectors ``A_i``.

    The momentum operator components for all spacial dimensions are
    calculated in parallel. [1]_

    References
    ----------
    .. [1] C. Klinkert, "The ab initio microscope: on the performance of
       2D materials as future field-effect transistors", Ph.D. thesis,
       ETH Zurich, 2021.

    """
    if tau_ij and centers is None:
        raise ValueError("Wannier centers needed if ``tau_ij`` is ``True``.")
    if approximate:
        pR_approximate = _approximate_momentum_operator(
            hR=hR,
            Ai=Ai,
            Ra=Ra,
            tau_ij=tau_ij,
            centers=centers,
            si_units=si_units,
        )
        return pR_approximate
    elif rR is None:
        raise ValueError("Position Matrix elements needed if ``approx`` is ``False``.")

    # NOTE: https://docs.python.org/3/library/pickle.html
    global _compute_pR_i

    # Spacial dimensions are treated in parallel.
    def _compute_pR_i(i: int) -> np.ndarray:
        """Computes the i-th spacial contribution to ``p_R``."""
        pR_i = np.zeros(hR.shape, dtype=np.complex64)
        d0_i = np.zeros(hR.shape[-2:])
        if tau_ij:
            # Trickery: Wannier center distances within cell from
            # transposed version of the Wannier centers themselves.
            d0_i = centers[:, i].reshape(-1, 1) - centers[:, i]
        # Iterate over all R vectors.
        for R in ndrange(pR_i.shape[:3], centered=True):
            allowed = np.any(hR[R])
            if Ra is not None:
                allowed = np.any(np.all(Ra == R, axis=1))
            if not allowed:
                continue
            pR_i[R] += hR[R] * ((R @ Ai)[i] + d0_i)
            # Iterate over all R' vectors.
            for Rp in ndrange(pR_i.shape[:3], centered=True):
                out_of_bounds = np.any(np.abs(R - Rp) > 0) or np.any(np.abs(R + Rp) > 0)
                allowed = np.any(hR[Rp]) and np.any(hR[(*(R - Rp),)])
                if Ra is not None:
                    allowed = np.any(np.all(Ra == Rp, axis=1)) and np.any(
                        np.all(Ra == (R - Rp), axis=1)
                    )
                if out_of_bounds or not allowed:
                    continue
                pR_i[R] += hR[Rp] @ rR[tuple(R - Rp)][..., i]
                pR_i[R] -= rR[Rp][..., i] @ hR[tuple(R - Rp)]
        return pR_i

    # Compute the spacial dimensions in parallel.
    pool = multiprocessing.Pool(3)
    _pR = pool.map(_compute_pR_i, range(3))

    # Put the momentum matrix together.
    pR = np.zeros(hR.shape + (3,), dtype=np.complex64)
    for i in range(pR.shape[-1]):
        pR[..., i] = _pR[i]

    # Conversion to SI units [kg*m/s].
    pR_SI = 1j * 1e-10 * m_e / hbar * pR
    if si_units:
        return pR_SI
    # Conversion to [eV/c].
    return c / e * pR_SI


def distance_matrix(
    Ai: np.ndarray,
    centers: np.ndarray,
    Ra: np.ndarray,
) -> np.ndarray:
    """Computes the Wannier center distance matrix.

    Parameters
    ----------
    Ai : np.ndarray
        Real-Space lattice vectors (3 x 3).
    centers : np.ndarray
        List of ``num_wann`` Wannier centers (``num_wann`` x 3) in real
        space.
    Ra : np.ndarray
        The allowed Wigner-Seitz cells to calculate the distance matrix
        for (``R_1``, ``R_2``, ``R_3``).

    Returns
    -------
    np.ndarray
        Matrix (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``) containing distances between Wannier centers for
        all the requested Wigner Seitz cells. The indices are chosen
        such that (0, 0, 0) actually gets you the center Wigner-Seitz
        cell distance matrix.

    """
    Ras = np.subtract(Ra, Ra.min(axis=0))
    N1, N2, N3 = Ras.max(axis=0) + 1

    # Trickery: Wannier center distances within cell from transposed
    # version of the Wannier centers themselves.
    d_0 = centers[:, np.newaxis] - centers

    # Midpoint of the Wigner-Seitz cell indices.
    num_wann = len(centers)
    d_R = np.zeros((N1, N2, N3, num_wann, num_wann))
    for R in ndrange((N1, N2, N3), centered=True):
        if np.any(np.all(Ra == R, axis=1)):
            d_R[R] = np.linalg.norm(d_0 + (R @ Ai), axis=2)

    return d_R


def k_sample(
    OR: np.ndarray,
    kpoints: np.ndarray = None,
    grid_size: tuple = None,
    einsum_optimize: Any = "optimal",
) -> np.ndarray:
    """Samples the given operator ``OR`` at given ``kpoints``.

    Parameters
    ----------
    OR : np.ndarray
        Operator elements (``N_1`` x ``N_2`` x ``N_3`` x ``num_wann`` x
        ``num_wann``).
    kpoints : np.ndarray, optional
        Reciprocal-space points in fractional coordinates (``N_k`` x 3).
        Used if ``monkhorst_pack`` is ``None``. This is ideal for path
        sampling.
    grid_size : tuple, optional
        Monkhorst-Pack grid size (``N_1``, ``N_2``, ``N_3``). Used if
        ``kpoints`` is ``None``.
    optimize : bool, optional
        Whether to optimize the einsum call. See ``np.einsum`` for more
        information.

    Returns
    -------
        Operator at the specified ``kpoints``.

    """
    if OR.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {OR.ndim=}")

    RR = np.zeros(OR.shape[:3] + (3,))
    for R in ndrange(OR.shape[:3], centered=True):
        RR[R] = R

    if kpoints is not None:
        if grid_size is None:
            raise ValueError("Either 'kpoints' or 'monkhorst_pack' must be specified.")

        R_dot_k = np.einsum("ijkr,lr->ijkl", RR, kpoints, optimize=einsum_optimize)
        phase = np.exp(2j * np.pi * R_dot_k)
        Ok = np.einsum("ijkmn,ijkl->lmn", OR, phase, optimize=einsum_optimize)
        return Ok

    grid_size = tuple(grid_size)
    kpoints = monkhorst_pack(grid_size)
    kR = np.zeros(grid_size + (3,))
    for R, kpoint in zip(ndrange(grid_size, centered=True), kpoints):
        kR[tuple(R)] = kpoint

    R_dot_k = np.einsum("ijkr,uvwr->ijkuvw", RR, kR, optimize=einsum_optimize)
    phase = np.exp(2j * np.pi * R_dot_k)
    Ok = np.einsum("ijkmn,ijkuvw->uvwmn", OR, phase, optimize=einsum_optimize)
    return Ok


def is_hermitian(OR: np.ndarray) -> bool:
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
    if OR.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {OR.ndim=}")

    for R in ndrange(OR.shape[:3], centered=True):
        if not spla.ishermitian(OR[R]):
            return False

    return True


def make_hermitian(OR: np.ndarray) -> np.ndarray:
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
    if OR.ndim != 5:
        raise ValueError(f"Inconsistent operator dimension: {OR.ndim=}")

    OR_hermitian = np.zeros_like(OR)
    for R in ndrange(OR.shape[:3], centered=True):
        OR_hermitian[R] = 0.5 * (np.conjugate(OR[R].T) + OR[-R])
        OR_hermitian[-R] = np.conjugate(OR_hermitian[R].T)

    return OR_hermitian
