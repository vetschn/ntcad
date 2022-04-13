#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" TODO: Docstrings.

"""
from scipy import constants

import numpy as np

hbar, *__ = constants.physical_constants["Planck constant over 2 pi in eV s"]
m_0 = constants.m_e


def momentum_operator(H_R: np.ndarray, r_R: np.ndarray) -> np.ndarray:
    """Calculates the momentum operator elements `p_R`.

    The momentum operator `p_R` is the commutator between Hamiltonian
    `H_R` and position operator `r_R` in the same Wannier basis.

    Note
    ----
    `N_i` correspond to the number of Wigner-Seitz cells along the
    lattice vectors `A_i`

    Parameters
    ----------
    H_R
        Hamiltonian elements (`N_1` x `N_2` x `N_3` x `num_wann` x
        `num_wann`).
    r_R
        Position matrix elements (`N_1` x `N_2` x `N_3` x `num_wann` x
        `num_wann` x 3).

    Returns
    -------
    p_R
        Momentum matrix elements (`N_1` x `N_2` x `N_3` x `num_wann` x
        `num_wann` x 3).

    """
    # Constant prefactor.
    c = 1j * m_0 / hbar

    # Iterate over R, R' and 3 spatial dimensions and populate the
    # momentum operator matrix with the commutator elements.
    # (Unfortunately ugly).
    p_R = np.zeros_like(r_R)
    for R_1, R_2, R_3 in np.ndindex(p_R.shape[:3]):
        for R_1_p, R_2_p, R_3_p in np.ndindex(R_1 + 1, R_2 + 1, R_3 + 1):
            for i in range(p_R.shape[-1]):
                H_R_R_p = H_R[R_1 - R_1_p, R_2 - R_2_p, R_3 - R_3_p]
                H_R_p = H_R[R_1_p, R_2_p, R_3_p]
                r_R_R_p_i = r_R[R_1 - R_1_p, R_2 - R_2_p, R_3 - R_3_p, ..., i]
                r_R_p_i = H_R[R_1_p, R_2_p, R_3_p, ..., i]
                p_R[R_1, R_2, R_3, ..., i] += c * (
                    H_R_R_p @ r_R_p_i - r_R_R_p_i @ H_R_p
                )
    return p_R


def approx_momentum_matrix():
    """_summary_
    TODO
    """
    pass


def distance_matrix(
    R_R: np.ndarray, A_i: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Computes the Wannier center distance matrix.

    Parameters
    ----------
    R_R
        Wigner-Seitz cell indices (`N_1` x `N_2` x `N_3` x 3).
    A_i
        Real-Space lattice vectors (3 x 3).
    centers
        List of `num_wann` Wannier centers (`num_wann` x 3) in real
        space.

    Returns
    -------
    d_R
        Matrix (`N_1` x `N_2` x `N_3` x `num_wann` x `num_wann`)
        containing distances between Wannier centers for all Wigner
        Seitz cells stored given by `R_R`.

    """
    num_wann = len(centers)

    # Trickery: Wannier center distances within cell from transposed
    # version of the Wannier centers themselves.
    d_0 = centers[:, np.newaxis] - centers

    d_R = np.zeros(R_R.shape[:-1] + (num_wann, num_wann))
    for R_1, R_2, R_3 in np.ndindex(R_R.shape[:-1]):
        R = R_R[R_1, R_2, R_3]
        d_R[R_1, R_2, R_3] = np.linalg.norm(d_0 + (A_i.T @ R), axis=2)

    return d_R


def k_sample(O_R: np.ndarray, R_R: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """Samples the given operator `O_R` at given `kpts`.

    Parameters
    ----------
    O_R
        Operator elements (`N_1` x `N_2` x `N_3` x `num_wann` x
        `num_wann`).
    R_R
        Wigner-Seitz cell indices (`N_1` x `N_2` x `N_3` x 3).
    kpts
        Reciprocal-space points in fractional coordinates (`N_k` x 3).

    Returns
    -------
    O_k
        Operator at the specified `kpts`.

    """
    phase = np.exp(2j * np.pi * np.einsum("ijkr,lr->ijkl", R_R, kpts))
    O_k = np.einsum("ijkmn,ijkr->rmn", O_R, phase)
    return O_k
