# -*- coding: utf-8 -*-
"""
Gaussian Process Prior
======================

Hello world
"""
# sphinx_gallery_thumbnail_number = 3

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# constants
num_features = 1  # dimensionality
num_index_points = 256  # nbr of index points
num_samples = 8

kernel = kernels.ExponentiatedQuadratic()
X_pred = np.linspace(-5.0, 5.0, num_index_points).reshape(-1, num_features)

seed = 23  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

# %%
# Kernel profile
# --------------
# The exponentiated quadratic kernel is *stationary*.
# That is, :math:`k(x, x') = k(x, 0)` for all :math:`x, x'`.

fig, ax = plt.subplots()

ax.plot(X_pred, kernel.apply(X_pred, np.zeros((1, num_features))))

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$k(x, 0)$')

plt.show()

# %%
# Kernel matrix
# -------------

fig, ax = plt.subplots()

ax.imshow(kernel.matrix(X_pred, X_pred), extent=[-5.0, 5.0, 5.0, -5.0],
          cmap="cividis")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$x$')

plt.show()

# %%
# Prior samples
# -------------

gp = tfd.GaussianProcess(kernel, X_pred)
samples = gp.sample(num_samples, seed=seed)

fig, ax = plt.subplots()

ax.plot(X_pred, samples.numpy().T)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')
ax.set_title(r'Draws of $f(x)$ from GP prior')

plt.show()
