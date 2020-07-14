"""Plotting module."""

import numpy as np
import matplotlib.pyplot as plt


def plot_image_grid(ax, images, shape, nrows=20, ncols=None, cmap=None):

    if ncols is None:
        ncols = nrows

    grid = images[:nrows*ncols].reshape(nrows, ncols, *shape).squeeze()

    return ax.imshow(np.vstack(np.dstack(grid)), cmap=cmap)


def fill_between_stddev(X_pred, mean_pred, stddev_pred, n=1, ax=None, *args,
                        **kwargs):

    if ax is None:
        ax = plt.gca()

    ax.fill_between(X_pred,
                    mean_pred - n * stddev_pred,
                    mean_pred + n * stddev_pred, **kwargs)
