#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" TODO: Docstrings.

"""
from typing import Callable
from matplotlib.colors import LogNorm, Normalize

import matplotlib.pyplot as plt
import numpy as np


def plot_operator(
    O_R: np.ndarray,
    axis: int = 2,
    indices: int = 0,
    mod: Callable = np.abs,
    norm: Normalize = LogNorm(),
    **kwargs: dict
) -> None:
    """_summary_

    Parameters
    ----------
    O_R
        _description_
    axis, optional
        _description_, by default 2
    indices, optional
        _description_, by default 0
    mod, optional
        _description_, by default np.abs
    """
    assert O_R.ndim == 5, "Inconsistent operator dimension."

    # Take operator blocks along one axis and apply modifier.
    O_ = np.take(mod(O_R), indices=indices, axis=axis)
    # Concatenate the operator Wigner-Seitz cell blocks together.
    O = np.concatenate([np.concatenate(block, axis=1) for block in O_], axis=0)

    # Plotting.
    ax = kwargs.get("ax")
    if ax is None:
        __, ax = plt.subplots()

    # The extent keyword here is used to set the ticks "correctly" and
    # to compensate for the fact that `ax.matshow` plots the pixels *on
    # top of* the index / coordinate.
    ax.matshow(O, norm=norm, extent=(0, O_.shape[0], O_.shape[1], 0), **kwargs)

    ticks = np.arange(0, O_.shape[0])
    ax.set_xticks(ticks, minor=True)
    ax.set_yticks(ticks, minor=True)
    ax.grid(which="both")


