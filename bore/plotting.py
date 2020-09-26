"""Plotting module."""

import numpy as np
import matplotlib.pyplot as plt


def fill_between_stddev(X_pred, mean_pred, stddev_pred, n=1, ax=None, *args,
                        **kwargs):

    if ax is None:
        ax = plt.gca()

    ax.fill_between(X_pred,
                    mean_pred - n * stddev_pred,
                    mean_pred + n * stddev_pred, **kwargs)
