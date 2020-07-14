# -*- coding: utf-8 -*-
"""
Vector Field Generative Art with Perlin noise
=============================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 5

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from noise import pnoise2

octaves = 5
factor = 3.0

num_lines = 600
num_steps = 50
step_sizes = 0.5 * (np.arange(10) + 1.0)

seed = 42  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

# resolution
h, w = 200, 200

y, x = np.ogrid[:h, :w]
X, Y = np.broadcast_arrays(x, y)

# %%
# The function ``pnoise2`` has range [-1, 1]. However, the outputs tend mostly
# to be centered around 0. Let's blow up the range by some ``factor`` and
# squash it through the :math:`\tanh` function so that it is still in the
# desired range.
Z = np.tanh(factor * np.vectorize(pnoise2)(x/w, y/h, octaves=octaves))  # range [-1, 1]
theta = np.pi * Z  # range [-pi, pi]

# %%
# Sparsify the grid so we can later draw intelligible quiver plots.
w_factor = h_factor = 10

x_sparse = x[..., ::w_factor]
y_sparse = y[::h_factor]

X_sparse = X[::h_factor, ::w_factor]
Y_sparse = Y[::h_factor, ::w_factor]

theta_sparse = theta[::w_factor, ::w_factor]

# %%
# We use use Perlin noise at every grid point as the angle (in radians).
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_title(r"angle $\theta$ (rad)")

contours = ax.pcolormesh(X, Y, theta, cmap="twilight")
fig.colorbar(contours, ax=ax)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()

# %%
# For sanity-check, we view the numerical values of the angles normalized by
# :math:`\pi` on the heatmap.
data = pd.DataFrame(theta_sparse / np.pi,
                    index=y_sparse.squeeze(axis=1),
                    columns=x_sparse.squeeze(axis=0))

# %%
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_title(r"angle $\theta / \pi$ (rad)")

sns.heatmap(data, annot=True, fmt=".2f", cmap="twilight", ax=ax)
ax.invert_yaxis()

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()

# %%
# Flow field
# ----------

dx = np.cos(theta_sparse)
dy = np.sin(theta_sparse)
# %%

fig, ax = plt.subplots(figsize=(10, 8))

ax.quiver(x_sparse, y_sparse, dx, dy)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()

# %%
# Path
# ----


# TODO: `w`, `h` arguments
def line(x, y, x_lim, y_lim, step_fn, num_steps, step_size=1.0):

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    if not (num_steps and x_min <= x < x_max and y_min <= y < y_max):
        return [], []

    dx, dy = step_fn(x, y, step_size=step_size)

    xs, ys = line(x=x+dx, y=y+dy, x_lim=x_lim, y_lim=y_lim, step_fn=step_fn,
                  num_steps=num_steps-1, step_size=step_size)
    xs.append(x)
    ys.append(y)

    return xs, ys


# %%
def step(x, y, step_size):

    j, i = int(x), int(y)

    dx = step_size * np.cos(theta[i, j])
    dy = step_size * np.sin(theta[i, j])

    return dx, dy


# %%
x = y = 25

xs, ys = line(x, y, x_lim=(0, w), y_lim=(0, h), step_fn=step,
              num_steps=num_steps, step_size=5.0)
_xs = np.asarray(xs[::-1])
_ys = np.asarray(ys[::-1])

# %%

fig, ax = plt.subplots(figsize=(10, 8))

ax.quiver(_xs[:-1], _ys[:-1], _xs[1:] - _xs[:-1], _ys[1:] - _ys[:-1],
          scale_units='xy', angles='xy', scale=1.0, width=3e-3, color='r')
ax.quiver(x_sparse, y_sparse, dx, dy)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()
# %%

dfs = []

for step_size in step_sizes:

    for lineno in range(num_lines):

        # sample starting point uniformly at random
        x = w * random_state.rand()
        y = h * random_state.rand()

        xs, ys = line(x, y, x_lim=(0, w), y_lim=(0, h), step_fn=step,
                      num_steps=num_steps, step_size=step_size)

        df = pd.DataFrame(dict(lineno=lineno, stepsize=step_size, x=xs, y=ys))
        dfs.append(df)
# %%

data = pd.concat(dfs, axis="index", sort=True)
data
# %%

fig, ax = plt.subplots(figsize=(10, 8))

sns.lineplot(x='x', y='y', hue='stepsize', units='lineno', estimator=None,
             sort=False, palette='Spectral', legend=None, linewidth=1.0,
             alpha=0.4, data=data, ax=ax)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()
