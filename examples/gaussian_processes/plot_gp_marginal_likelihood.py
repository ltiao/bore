# -*- coding: utf-8 -*-
"""
Gaussian Process Marginal Likelihood
====================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from etudes.datasets import synthetic_sinusoidal, make_regression_dataset
from etudes.plotting import fill_between_stddev
from etudes.gaussian_process import dataframe_from_gp_summary
# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# constants
num_train = 25  # nbr training points in synthetic dataset
num_features = 1  # dimensionality
num_index_points = 256  # nbr of index points
num_samples = 7

observation_noise_variance = 1e-1
jitter = 1e-6

kernel_cls = kernels.ExponentiatedQuadratic

seed = 42  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

golden_ratio = 0.5 * (1 + np.sqrt(5))

X_pred = np.linspace(-1.0, 1.0, num_index_points).reshape(-1, num_features)

load_data = make_regression_dataset(synthetic_sinusoidal)
X_train, Y_train = load_data(num_train, num_features,
                             observation_noise_variance,
                             x_min=-0.5, x_max=0.5,
                             random_state=random_state)

# %%
# Synthetic dataset
# -----------------

fig, ax = plt.subplots()

ax.plot(X_pred, synthetic_sinusoidal(X_pred), label="true")
ax.scatter(X_train, Y_train, marker='x', color='k',
           label="noisy observations")

ax.legend()

ax.set_xlim(-0.6, 0.6)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()

# %%
# Log marginal likelihood (LML)
# -----------------------------

amplitude, length_scale = np.ogrid[5e-2:4.0:100j, 1e-5:5e-1:100j]
kernel = kernel_cls(amplitude, length_scale)

gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=X_train,
    observation_noise_variance=observation_noise_variance
)

nll = - gp.log_prob(Y_train)

# %%
# 3D Surface Plot
# ^^^^^^^^^^^^^^^

fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=25, elev=35))

ax.plot_surface(amplitude, length_scale, nll,  # rstride=1, cstride=1,
                edgecolor='k', linewidth=0.5, cmap="Spectral_r")

ax.set_xlabel(r"amplitude $\sigma$")
ax.set_ylabel(r"lengthscale $\ell$")
ax.set_zlabel("negative log marginal likelihood")

plt.show()

# %%
# Contour Plot
# ^^^^^^^^^^^^

amplitude_grid, length_scale_grid = amplitude[10::20], length_scale[..., 10::20]
kernel_grid = kernel_cls(amplitude_grid, length_scale_grid)

theta = np.dstack(np.broadcast_arrays(amplitude_grid, length_scale_grid)).reshape(-1, 2)

fig, ax = plt.subplots()

ax.scatter(*theta.T, color='k', marker='+')
contours = ax.contour(*np.broadcast_arrays(amplitude, length_scale), nll,
                      cmap="Spectral_r")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt='%.1f')

ax.set_xlabel(r"amplitude $\sigma$")
ax.set_ylabel(r"lengthscale $\ell$")

plt.show()

# %%
# Posterior predictive distributions
# ----------------------------------


def scatterplot(X, Y, ax=None, *args, **kwargs):

    if ax is None:
        ax = plt.gca()

    ax.scatter(X, Y, *args, **kwargs)


gprm_grid = tfd.GaussianProcessRegressionModel(
    kernel=kernel_grid, index_points=X_pred,
    observation_index_points=X_train, observations=Y_train,
    observation_noise_variance=observation_noise_variance,
    jitter=jitter
)

data = dataframe_from_gp_summary(gprm_grid.mean().numpy(),
                                 gprm_grid.stddev().numpy(),
                                 amplitude_grid, length_scale_grid, X_pred) \
    .rename(columns={"length_scale": "lengthscale", "index_point": "x"})

g = sns.relplot(x="x", y="mean",
                row="amplitude", col="lengthscale",
                row_order=amplitude_grid[::-1].squeeze(),
                height=5.0, aspect=golden_ratio, kind="line",
                data=data, alpha=0.7, linewidth=3.0)
g.map(scatterplot, X=X_train, Y=Y_train, s=8.0**2, marker='x', color='k')
g.map(fill_between_stddev, "x", "mean", "stddev", alpha=0.1)
g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
             col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
g.set_axis_labels(r"$x$", r"$y$")
g.set(ylim=(-2.0, 1.5))
