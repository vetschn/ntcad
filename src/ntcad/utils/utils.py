#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" TODO: Docstrings.

"""
import numpy as np


def monkhorst_pack(size: np.ndarray):
    """Constructs a uniform sampling of k-space of given size.

    Parameters
    ----------
    size
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpts + 0.5) / size - 0.5


def k_path(points: np.ndarray, num: int = 50) -> np.ndarray:
    """Generates a k-point path along the given symmetry points.

    Parameters
    ----------
    points
        Symmetry points along the path (``N_p`` x 3), where ``N_p`` is the
        number of symmetry points.
    num, optional
        The number of k-points along each section, by default 50.

    Returns
    -------
    kpts
        All k-points along the given symmetry points (``N_s``*``num`` x 3),
        where ``N_s`` is the number of sections between symmetry points.
    """
    N_s = len(points) - 1
    sections = np.zeros((N_s, num, 3))
    for i in range(N_s):
        sections[i] = np.linspace(points[i], points[i + 1], num)
    kpts = sections.reshape((N_s * num, 3))
    return kpts
