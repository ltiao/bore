# -*- coding: utf-8 -*-
"""
Gaussian Process with Period Kernels
====================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 5

import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns

from etudes.gaussian_process import gp_sample_custom, dataframe_from_gp_samples

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# constants
num_features = 1  # dimensionality
num_index_points = 256  # nbr of index points
num_samples = 8
x_min, x_max = -np.pi, np.pi
period = np.float64(2. * np.pi)

X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)

seed = 23  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

kernel_cls = kernels.ExpSinSquared
kernel = kernel_cls(period=period)
# %%
# Kernel profile
# --------------
# The exponentiated quadratic kernel is *stationary*.
# That is, :math:`k(x, x') = k(x, 0)` for all :math:`x, x'`.

fig, ax = plt.subplots()

ax.plot(X_grid, kernel.apply(X_grid, np.zeros((1, num_features))))

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$k(x, 0)$')

plt.show()

# %%
# Kernel matrix
# -------------
x1, x2 = np.broadcast_arrays(X_grid, X_grid.T)
# %%

fig, ax = plt.subplots()

ax.pcolormesh(x1, x2, kernel.matrix(X_grid, X_grid), cmap="cividis")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$x$')

plt.show()

# %%
# Prior samples
# -------------

gp = tfd.GaussianProcess(kernel=kernel, index_points=X_grid)
samples = gp.sample(num_samples, seed=seed)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, samples.numpy().T)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')
ax.set_title(r'Draws of $f(x)$ from GP prior')

plt.show()
# %%

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

ax.plot(X_grid, samples.numpy().T)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')
ax.set_title(r'Draws of $f(x)$ from GP prior')

plt.show()
# %%

amplitude, length_scale_inv = np.ogrid[1.5:3.6:2j, 10.0:0.5:3j]
length_scale = 1.0 / length_scale_inv
# %%

kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale, period=period)
gp = tfd.GaussianProcess(kernel=kernel, index_points=X_grid)
gp_samples = gp_sample_custom(gp, num_samples, seed=seed)
# %%

data = dataframe_from_gp_samples(gp_samples.numpy(), X_grid, amplitude,
                                 length_scale, num_samples)
data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)
# %%

g = sns.relplot(x="index point", y="function value", hue="sample",
                row="amplitude", col="length scale", height=5.0, aspect=1.0,
                kind="line", data=data, alpha=0.7, linewidth=3.0,
                facet_kws=dict(subplot_kws=dict(projection='polar'), despine=False))
g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
             col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
g.set_axis_labels(r"$x$", r"$f^{(i)}(x)$")
