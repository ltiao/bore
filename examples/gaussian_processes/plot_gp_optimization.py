# -*- coding: utf-8 -*-
"""
GP Hyperparameter Estimation
============================

Here we fit the hyperparameters of a Gaussian Process by maximizing the (log)
marginal likelihood. This is commonly referred to as empirical Bayes, or
type-II maximum likelihood estimation.
"""
# sphinx_gallery_thumbnail_number = 3

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import defaultdict
from etudes.datasets import synthetic_sinusoidal, make_regression_dataset
from etudes.plotting import fill_between_stddev

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


def to_numpy(transformed_variable):

    return tf.convert_to_tensor(transformed_variable).numpy()


# constants
num_train = 25  # nbr training points in synthetic dataset
num_features = 1  # dimensionality
num_index_points = 256  # nbr of index points
num_samples = 7

num_epochs = 200

observation_noise_variance_true = 1e-1
jitter = 1e-6

kernel_cls = kernels.ExponentiatedQuadratic

seed = 42  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

golden_ratio = 0.5 * (1 + np.sqrt(5))

X_pred = np.linspace(-1.0, 1.0, num_index_points).reshape(-1, num_features)

load_data = make_regression_dataset(synthetic_sinusoidal)
X_train, Y_train = load_data(num_train, num_features,
                             observation_noise_variance_true,
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

amplitude = tfp.util.TransformedVariable(
    1.0, bijector=tfp.bijectors.Exp(), dtype="float64", name='amplitude')
length_scale = tfp.util.TransformedVariable(
    0.5, bijector=tfp.bijectors.Exp(), dtype="float64", name='length_scale')
observation_noise_variance = tfp.util.TransformedVariable(
    1e-1, bijector=tfp.bijectors.Exp(), dtype="float64",
    name='observation_noise_variance')

# %%

kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)
gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=X_train,
    observation_noise_variance=observation_noise_variance)

# %%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.5,
                                     beta_2=0.99)

# %%


history = defaultdict(list)

for epoch in range(num_epochs):

    with tf.GradientTape() as tape:
        nll = - gp.log_prob(Y_train)

    gradients = tape.gradient(nll, gp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gp.trainable_variables))

    history["nll"].append(to_numpy(nll))
    history["amplitude"].append(to_numpy(amplitude))
    history["length_scale"].append(to_numpy(length_scale))
    history["observation_noise_variance"].append(to_numpy(observation_noise_variance))

# %%

amplitude_grid, length_scale_grid = np.ogrid[5e-2:4.0:100j, 1e-5:5e-1:100j]
kernel_grid = kernel_cls(amplitude=amplitude_grid,
                         length_scale=length_scale_grid)
gp_grid = tfd.GaussianProcess(
    kernel=kernel_grid,
    index_points=X_train,
    observation_noise_variance=observation_noise_variance_true)
nll_grid = - gp_grid.log_prob(Y_train)

# %%

fig, ax = plt.subplots()

contours = ax.contour(*np.broadcast_arrays(amplitude_grid, length_scale_grid),
                      nll_grid, cmap="Spectral_r")

sns.lineplot(x='amplitude', y='length_scale',
             sort=False, data=pd.DataFrame(history), alpha=0.8, ax=ax)

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt='%.1f')

ax.set_xlabel(r"amplitude $\sigma$")
ax.set_ylabel(r"lengthscale $\ell$")

plt.show()

# %%

kernel_history = kernel_cls(amplitude=history["amplitude"],
                            length_scale=history["length_scale"])
gprm_history = tfd.GaussianProcessRegressionModel(
    kernel=kernel_history, index_points=X_pred,
    observation_index_points=X_train, observations=Y_train,
    observation_noise_variance=history["observation_noise_variance"],
    jitter=jitter)
gprm_mean = gprm_history.mean()
gprm_stddev = gprm_history.stddev()

# %%

# "Melt" the dataframe
d = pd.DataFrame(gprm_mean.numpy(), columns=X_pred.squeeze())
d.index.name = "epoch"
d.columns.name = "x"
s = d.stack()
s.name = "y"
data = s.reset_index()
data

# %%

fig, ax = plt.subplots()

sns.lineplot(x='x', y='y', hue="epoch", palette="viridis_r", data=data,
             linewidth=0.2, ax=ax)
ax.scatter(X_train, Y_train, marker='x', color='k', label="noisy observations")

ax.set_xlabel('$x$')
ax.set_ylabel('$\mu(x)$')

plt.show()

# %%

# "Melt" the dataframe
d = pd.DataFrame(gprm_stddev.numpy(), columns=X_pred.squeeze())
d.index.name = "epoch"
d.columns.name = "x"
s = d.stack()
s.name = "y"
data = s.reset_index()
data

# %%

fig, ax = plt.subplots()

sns.lineplot(x='x', y='y', hue="epoch", palette="viridis_r", data=data,
             linewidth=0.2, ax=ax)

ax.set_xlabel('$x$')
ax.set_ylabel('$\sigma(x)$')

plt.show()
