# -*- coding: utf-8 -*-
"""
Bayesian Optimization as Density Ratio Estimation
=================================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 1
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from bore.datasets import make_regression_dataset
# %%


def latent(gamma):
    """
    Forrester's.
    """
    # return (6.0*gamma-2.0)**2 * np.sin(12.0 * gamma - 4.0)
    return np.sin(3.0*gamma) + gamma**2 - 0.7*gamma
# %%


# constants
num_features = 1
num_init_random = 64
noise_variance = 0.2
gamma = 1/3
bandwidth = 0.25

num_index_points = 512
x_min, x_max = -1.0, 2.0

seed = 8888  # set random seed for reproducibility
random_state = np.random.RandomState(seed)
# %%

X = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)
y = latent(X)
# %%

fig, ax = plt.subplots()

ax.plot(X, y, c="tab:gray")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r"$y$ (test mse)")

plt.show()

# %%

load_observations = make_regression_dataset(latent)
X_samples, y_samples = load_observations(num_samples=num_init_random,
                                         num_features=num_features,
                                         noise_variance=noise_variance,
                                         x_min=x_min, x_max=x_max,
                                         random_state=random_state)
# %%

tau = np.quantile(y_samples, q=gamma)
mask_l = np.less(y_samples, tau)
mask_g = ~mask_l

X_samples_l = X_samples[mask_l]
X_samples_g = X_samples[mask_g]

y_samples_l = y_samples[mask_l]
y_samples_g = y_samples[mask_g]
# %%

y_samples_sorted = np.sort(y_samples)
y_samples_quantile = np.arange(num_init_random) / num_init_random
# %%


fig, ax = plt.subplots()

ax.plot(X, y, c="tab:gray")

ax.scatter(X_samples_l, y_samples_l, alpha=0.8)
ax.scatter(X_samples_g, y_samples_g, alpha=0.8)

ax.axhline(tau, xmin=0, xmax=1.0, color='k', linewidth=1.0, linestyle='dashed')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r"$y$ (test mse)")

plt.show()
# %%

fig, ax = plt.subplots()

sns.ecdfplot(x=y_samples, ax=ax)

ax.axvline(tau, ymin=0., ymax=gamma,
           color="black", linestyle='dashed', linewidth=1.0)
ax.hlines(gamma, xmin=y_samples.min(), xmax=tau,
          colors="black", linestyles='dashed', linewidth=1.0)

ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$\Phi(y)$')

plt.show()
# %%

fig, ax = plt.subplots()

sns.kdeplot(x=X_samples_l.ravel(), fill=True, bw_method=bandwidth,
            label=r'$\ell(x)$', ax=ax)
sns.kdeplot(x=X_samples_g.ravel(), fill=True, bw_method=bandwidth,
            label=r'$g(x)$', ax=ax)

sns.rugplot(X_samples_l.ravel(), ax=ax)
sns.rugplot(X_samples_g.ravel(), ax=ax)

ax.set_xlabel(r'$x$')
ax.set_ylabel("density")

ax.legend()

plt.show()
# %%
frame = pd.DataFrame(data=X_samples,
                     columns=['x']).assign(y=y_samples, z=mask_l)
frame
# %%
g = sns.displot(data=frame, x='x', hue='z', rug=True, kind="kde",
                fill=True, bw_method=bandwidth)
# %%
g = sns.JointGrid(height=6, ratio=2, space=.05, marginal_ticks=True)

g.ax_joint.plot(X, y, c="tab:gray")
sns.scatterplot(data=frame, x='x', y='y', hue='z', ax=g.ax_joint)
g.ax_joint.axhline(tau, xmin=0., xmax=1., color='k',
                   linewidth=1.0, linestyle='dashed')

sns.ecdfplot(data=frame, y='y', ax=g.ax_marg_y)
g.ax_marg_y.vlines(gamma, ymin=y_samples.min(), ymax=tau, colors="black",
                   linestyles='dashed', linewidth=1.)
g.ax_marg_y.axhline(tau, xmin=0., xmax=gamma, color="black",
                    linestyle='dashed', linewidth=1.)

sns.kdeplot(data=frame, x='x', hue='z', fill=True, legend=False, ax=g.ax_marg_x)
sns.rugplot(data=frame, x='x', hue='z', legend=False, ax=g.ax_marg_x)
