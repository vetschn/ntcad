"""
Useful processing routines and operations for CP2K data.

"""

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb, factorial

from ntcad.utils import get_tuples_summing_to


def primitive_gto(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    l_x: int,
    l_y: int,
    l_z: int,
    a_i: float,
) -> np.ndarray | float:
    """Evaluates a primitive Gaussian-type orbital.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the orbital.
    y : array_like
        The y-coordinates at which to evaluate the orbital.
    z : array_like
        The z-coordinates at which to evaluate the orbital.
    l_x : int
        The angular momentum in the x-direction.
    l_y : int
        The angular momentum in the y-direction.
    l_z : int
        The angular momentum in the z-direction.
    a_i : float
        The exponent of the Gaussian.

    Returns
    -------
    primitive_gto : ndarray or float
        The evaluated primitive Gaussian-type orbital.

    Examples
    --------
    Evaluate a 2p orbital at the origin:
    >>> primitive_gto(0, 0, 0, 1, 1, 1, 1)
    0.0

    """
    return x**l_x * y**l_y * z**l_z * np.exp(-a_i * (x**2 + y**2 + z**2))


def normalized_gto(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    l_x: int,
    l_y: int,
    l_z: int,
    a_i: float,
) -> np.ndarray | float:
    """Evaluates a normalized Gaussian-type orbital.

    This computes a normalization factor and multiplies the primitive
    Gaussian-type orbital by it.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the orbital.
    y : array_like
        The y-coordinates at which to evaluate the orbital.
    z : array_like
        The z-coordinates at which to evaluate the orbital.
    l_x : int
        The angular momentum in the x-direction.
    l_y : int
        The angular momentum in the y-direction.
    l_z : int
        The angular momentum in the z-direction.
    a_i : float
        The exponent of the Gaussian.

    Returns
    -------
    normalized_gto : ndarray or float
        The evaluated normalized Gaussian-type orbital.

    """
    norm_factor = (2 * a_i / np.pi) ** (3 / 4)
    norm_factor *= np.sqrt((8 * a_i) ** (l_x + l_y + l_z))
    norm_factor *= np.sqrt(factorial(l_x) * factorial(l_y) * factorial(l_z))
    norm_factor /= np.sqrt(factorial(2 * l_x) * factorial(2 * l_y) * factorial(2 * l_z))

    return norm_factor * primitive_gto(x, y, z, l_x, l_y, l_z, a_i)


def contracted_gto(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    l_x: int,
    l_y: int,
    l_z: int,
    a: ArrayLike,
    c: ArrayLike,
) -> np.ndarray | float:
    """Evaluates a normalized contracted Gaussian-type orbital.

    This computes a contraction normalization factor and multiplies the
    normalized Gaussian-type orbital by it.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the orbital.
    y : array_like
        The y-coordinates at which to evaluate the orbital.
    z : array_like
        The z-coordinates at which to evaluate the orbital.
    l_x : int
        The angular momentum in the x-direction.
    l_y : int
        The angular momentum in the y-direction.
    l_z : int
        The angular momentum in the z-direction.
    a : array_like
        The exponents of the Gaussians.
    c : array_like
        The contraction coefficients.

    Returns
    -------
    contracted_gto : ndarray or float
        The evaluated normalized contracted Gaussian-type orbital.

    """
    l = l_x + l_y + l_z

    norm_factor = sum(
        [
            c_i * c_j * (2 * np.sqrt(a_i * a_j) / (a_i + a_j)) ** (l + 1.5)
            for a_i, c_i in zip(a, c)
            for a_j, c_j in zip(a, c)
        ]
    )
    g = np.sum(
        [normalized_gto(x, y, z, l_x, l_y, l_z, a_i) * c_i for a_i, c_i in zip(a, c)],
        axis=0,
    )
    return g / np.sqrt(norm_factor)


def cart_to_pure_coeff(l: int, m: int, l_x: int, l_y: int, l_z: int) -> complex:
    """Computes the contribution from cart. to pure sph. harmonic GTO.

    Parameters
    ----------
    l : int
        The angular momentum quantum number.
    m : int
        The magnetic quantum number.
    l_x : int
        The angular momentum in the x-direction.
    l_y : int
        The angular momentum in the y-direction.
    l_z : int
        The angular momentum in the z-direction.

    Returns
    -------
    coeff : complex
        The coefficient for the contribution from the Cartesian to the
        pure spherical harmonic Gaussian-type orbital.

    """
    j = (l_x + l_y - abs(m)) / 2

    if j < 0 or j != int(j):
        return 0.0
    else:
        j = int(j)

    numerator = (
        factorial(2 * l_x)
        * factorial(2 * l_y)
        * factorial(2 * l_z)
        * factorial(l)
        * factorial(l - abs(m))
    )
    denominator = (
        factorial(2 * l)
        * factorial(l_x)
        * factorial(l_y)
        * factorial(l_z)
        * factorial(l + abs(m))
    )
    first_term = np.emath.sqrt(numerator / denominator)

    second_term_prefactor = 1 / (2**l * factorial(l))
    second_term_sum = 0
    for i in range((l - abs(m)) // 2 + 1):
        temp_0 = comb(l, i) * comb(i, j)
        temp_1 = (-1) ** i * factorial(2 * l - 2 * i) / factorial(l - abs(m) - 2 * i)
        temp_sum = sum(
            (comb(j, k) * comb(abs(m), l_x - 2 * k))
            * np.emath.sqrt((-1) ** float(np.sign(m) * (abs(m) - l_x + 2 * k)))
            for k in range(j + 1)
        )
        second_term_sum += temp_0 * temp_1 * temp_sum

    second_term = second_term_prefactor * second_term_sum

    return first_term * second_term


def pure_sph_harm_gto(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    l: int,
    m: int,
    a: ArrayLike,
    c: ArrayLike,
) -> np.ndarray | float:
    """Evaluates a pure spherical harmonic contracted GTO.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the orbital.
    y : array_like
        The y-coordinates at which to evaluate the orbital.
    z : array_like
        The z-coordinates at which to evaluate the orbital.
    l : int
        The angular momentum quantum number.
    m : int
        The magnetic quantum number.
    a : array_like
        The exponents of the Gaussians.
    c : array_like
        The contraction coefficients.

    Returns
    -------
    pure_sph_harm_gto : ndarray or float
        The evaluated pure spherical harmonic contracted Gaussian-type
        orbital.

    """
    l_tuples = list(get_tuples_summing_to(l))
    # These are always already real.
    coeff = [cart_to_pure_coeff(l, m, *l_tuple) for l_tuple in l_tuples]
    psi = np.sum(
        (
            contracted_gto(x, y, z, *l_tuple, a, c) * w
            for l_tuple, w in zip(l_tuples, coeff)
            if w != 0
        ),
        axis=0,
    )
    if m == 0:
        return psi

    coeff_m = [cart_to_pure_coeff(l, -m, *l_tuple) for l_tuple in l_tuples]
    psi_m = np.sum(
        [
            contracted_gto(x, y, z, *l_tuple, a, c) * w
            for l_tuple, w in zip(l_tuples, coeff_m)
            if w != 0
        ],
        axis=0,
    )
    if m > 0:
        return (psi + psi_m) / np.sqrt(2)

    return (psi - psi_m) / np.sqrt(2)


def extract_basis_functions(basis_set: dict) -> list[Callable]:
    """Extracts all the basis functions from a basis set.

    Parameters
    ----------
    basis_set : dict
        The basis set read from a basis set database file.

    Yields
    ------
    basis_function : callable
        A callable that evaluates a basis function.

    See Also
    --------
    ntcad.cp2k.io.read_basis_set_database

    """
    basis_functions = []
    for param_set in basis_set["param_sets"]:
        a = param_set["a"]
        for l in range(param_set["l_min"], param_set["l_max"] + 1):
            for m in range(-l, l + 1):
                for c in param_set["c"][l].T:
                    basis_functions.append(
                        lambda x, y, z: pure_sph_harm_gto(x, y, z, l, m, a, c)
                    )

    return basis_functions
