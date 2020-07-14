# -*- coding: utf-8 -*-
"""
Variational Sparse GP for Binary Classification
===============================================

Here we fit the hyperparameters of a Gaussian Process by maximizing the (log)
marginal likelihood. This is commonly referred to as empirical Bayes, or
type-II maximum likelihood estimation.
"""
# sphinx_gallery_thumbnail_number = 8

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.layers import Layer, InputLayer
from tensorflow.keras.initializers import Identity, Constant

from etudes.datasets import make_classification_dataset
from etudes.plotting import fill_between_stddev
from etudes.utils import get_kl_weight

from collections import defaultdict

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


# constants
num_train = 2048  # nbr training points in synthetic dataset
num_test = 40
num_features = 1  # dimensionality
num_index_points = 256  # nbr of index points
num_samples = 25
quadrature_size = 20

num_inducing_points = 50
num_epochs = 1000
batch_size = 64
shuffle_buffer_size = 500

jitter = 1e-6

kernel_cls = kernels.MaternFiveHalves

seed = 8888  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

x_min, x_max = -5.0, 5.0
y_min, y_max = -6.0, 4.0

# index points
X_q = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)

golden_ratio = 0.5 * (1 + np.sqrt(5))

# %%
# Synthetic dataset
# -----------------

p = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
    components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
q = tfd.Normal(loc=0.0, scale=2.0)

# %%


def load_data(num_samples, rate=0.5, dtype="float64", seed=None):

    num_p = int(num_samples * rate)
    num_q = num_samples - num_p

    X_p = p.sample(sample_shape=(num_p, 1), seed=seed).numpy()
    X_q = q.sample(sample_shape=(num_q, 1), seed=seed).numpy()

    X, y = make_classification_dataset(X_p, X_q, dtype=dtype)

    return X, y

# %%


def logit(x):

    return p.log_prob(x) - q.log_prob(x)

# %%


def density_ratio(x):

    return tf.exp(logit(x))

# %%


def optimal_score(x):

    return tf.sigmoid(logit(x))

# %%
# Probability densities


fig, ax = plt.subplots()

ax.plot(X_q, q.prob(X_q), label='$q(x)$')
ax.plot(X_q, p.prob(X_q), label='$p(x)$')

ax.set_xlabel('$x$')
ax.set_ylabel('density')

ax.legend()

plt.show()

# %%
# Log density ratio, log-odds, or logits.

fig, ax = plt.subplots()

ax.plot(X_q, logit(X_q), c='k', label=r"$f(x) = \log p(x) - \log q(x)$")

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')

ax.legend()

plt.show()

# %%
# Density ratio.

fig, ax = plt.subplots()

ax.plot(X_q, density_ratio(X_q), c='k', label=r"$r(x) = \exp f(x)$")

ax.set_xlabel('$x$')
ax.set_ylabel('$r(x)$')

ax.legend()

plt.show()

# %%
# Create classification dataset.

X_train, y_train = load_data(num_train, seed=seed)
X_test, y_test = load_data(num_test, seed=seed)

# %%
# Dataset visualized against the Bayes optimal classifier.

fig, ax = plt.subplots()

ax.plot(X_q, optimal_score(X_q), c='k', label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X_train, y_train, c=y_train, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()

# %%
# Encapsulate Variational Gaussian Process (particular variable initialization)
# in a Keras / TensorFlow Probability Mixin Layer.
# Clean and simple if we restrict to single-output (`event_shape = ()`) and
# `feature_ndim = 1` (i.e. inputs are simply vectors rather than matrices or
# tensors).


class VariationalGaussianProcess1D(tfp.layers.DistributionLambda):

    def __init__(self, kernel_wrapper, num_inducing_points,
                 inducing_index_points_initializer, mean_fn=None, jitter=1e-6,
                 convert_to_tensor_fn=tfd.Distribution.sample, **kwargs):

        def make_distribution(x):

            return VariationalGaussianProcess1D.new(
                x, kernel_wrapper=self.kernel_wrapper,
                inducing_index_points=self.inducing_index_points,
                variational_inducing_observations_loc=(
                    self.variational_inducing_observations_loc),
                variational_inducing_observations_scale=(
                    self.variational_inducing_observations_scale),
                mean_fn=self.mean_fn,
                observation_noise_variance=tf.exp(
                    self.log_observation_noise_variance),
                jitter=self.jitter)

        super(VariationalGaussianProcess1D, self).__init__(
            make_distribution_fn=make_distribution,
            convert_to_tensor_fn=convert_to_tensor_fn,
            dtype=kernel_wrapper.dtype)

        self.kernel_wrapper = kernel_wrapper
        self.inducing_index_points_initializer = inducing_index_points_initializer
        self.num_inducing_points = num_inducing_points
        self.mean_fn = mean_fn
        self.jitter = jitter

        self._dtype = self.kernel_wrapper.dtype

    def build(self, input_shape):

        input_dim = input_shape[-1]

        # TODO: Fix initialization!
        self.inducing_index_points = self.add_weight(
            name="inducing_index_points",
            shape=(self.num_inducing_points, input_dim),
            initializer=self.inducing_index_points_initializer,
            dtype=self.dtype)

        self.variational_inducing_observations_loc = self.add_weight(
            name="variational_inducing_observations_loc",
            shape=(self.num_inducing_points,),
            initializer="zeros", dtype=self.dtype)

        self.variational_inducing_observations_scale = self.add_weight(
            name="variational_inducing_observations_scale",
            shape=(self.num_inducing_points, self.num_inducing_points),
            initializer=Identity(gain=1.0), dtype=self.dtype)

        self.log_observation_noise_variance = self.add_weight(
            name="log_observation_noise_variance",
            initializer=Constant(-5.0), dtype=self.dtype)

    @staticmethod
    def new(x, kernel_wrapper, inducing_index_points, mean_fn,
            variational_inducing_observations_loc,
            variational_inducing_observations_scale,
            observation_noise_variance, jitter, name=None):

        # ind = tfd.Independent(base, reinterpreted_batch_ndims=1)
        # bijector = tfp.bijectors.Transpose(rightmost_transposed_ndims=2)
        # d = tfd.TransformedDistribution(ind, bijector=bijector)

        return tfd.VariationalGaussianProcess(
            kernel=kernel_wrapper.kernel, index_points=x,
            inducing_index_points=inducing_index_points,
            variational_inducing_observations_loc=(
                variational_inducing_observations_loc),
            variational_inducing_observations_scale=(
                variational_inducing_observations_scale),
            mean_fn=mean_fn,
            observation_noise_variance=observation_noise_variance,
            jitter=jitter)

# %%
# Kernel wrapper layer


class KernelWrapper(Layer):

    # TODO: Support automatic relevance determination
    def __init__(self, kernel_cls=kernels.ExponentiatedQuadratic,
                 dtype=None, **kwargs):

        super(KernelWrapper, self).__init__(dtype=dtype, **kwargs)

        self.kernel_cls = kernel_cls

        self.log_amplitude = self.add_weight(
            name="log_amplitude",
            initializer="zeros", dtype=dtype)

        self.log_length_scale = self.add_weight(
            name="log_length_scale",
            initializer="zeros", dtype=dtype)

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return self.kernel_cls(amplitude=tf.exp(self.log_amplitude),
                               length_scale=tf.exp(self.log_length_scale))

# %%
# Bernoulli likelihood for binary classification.


def make_binary_classification_likelihood(f):

    return tfd.Independent(tfd.Bernoulli(logits=f),
                           reinterpreted_batch_ndims=1)
# %%


def log_likelihood(y, f):

    likelihood = make_binary_classification_likelihood(f)
    return likelihood.log_prob(y)

# %%
# Helper Model factory method.


def build_model(input_dim, jitter=1e-6):

    inducing_index_points_initial = random_state.choice(X_train.squeeze(),
                                                        num_inducing_points) \
                                                .reshape(-1, num_features)

    inducing_index_points_initializer = (
        tf.constant_initializer(inducing_index_points_initial))

    return tf.keras.Sequential([
        InputLayer(input_shape=(input_dim,)),
        VariationalGaussianProcess1D(
            kernel_wrapper=KernelWrapper(kernel_cls=kernel_cls,
                                         dtype=tf.float64),
            num_inducing_points=num_inducing_points,
            inducing_index_points_initializer=inducing_index_points_initializer,
            jitter=jitter)
    ])
# %%


model = build_model(input_dim=num_features, jitter=jitter)
optimizer = tf.keras.optimizers.Adam()
# %%


@tf.function
def nelbo(X_batch, y_batch):

    qf = model(X_batch)

    ell = qf.surrogate_posterior_expected_log_likelihood(
        observations=y_batch,
        log_likelihood_fn=log_likelihood,
        quadrature_size=quadrature_size)

    kl = qf.surrogate_posterior_kl_divergence_prior()
    kl_weight = get_kl_weight(num_train, batch_size)

    return - ell + kl_weight * kl
# %%


@tf.function
def train_step(X_batch, y_batch):

    with tf.GradientTape() as tape:
        loss = nelbo(X_batch, y_batch)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
# %%


dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                         .shuffle(seed=seed, buffer_size=shuffle_buffer_size) \
                         .batch(batch_size, drop_remainder=True)
# %%

keys = ["inducing_index_points",
        "variational_inducing_observations_loc",
        "variational_inducing_observations_scale",
        "log_observation_noise_variance",
        "log_amplitude", "log_length_scale"]
# %%

history = defaultdict(list)

for epoch in range(num_epochs):

    for step, (X_batch, y_batch) in enumerate(dataset):

        loss = train_step(X_batch, y_batch)

    print("epoch={epoch:04d}, loss={loss:.4f}"
          .format(epoch=epoch, loss=loss.numpy()))

    history["nelbo"].append(loss.numpy())

    for key, tensor in zip(keys, model.get_weights()):

        history[key].append(tensor)


# %%
# Create test-time model with higher `jitter` to be robust to unseen test
# inputs.
test_model = build_model(input_dim=num_features, jitter=2.0*jitter)
test_model.set_weights(model.get_weights())

# %%

inducing_index_points_history = history.pop("inducing_index_points")
variational_inducing_observations_loc_history = (
    history.pop("variational_inducing_observations_loc"))

inducing_index_points = inducing_index_points_history[-1]
variational_inducing_observations_loc = (
    variational_inducing_observations_loc_history[-1])

# %%
# Log density ratio, log-odds, or logits.

fig, ax = plt.subplots()

ax.plot(X_q, logit(X_q), c='k',
        label=r"$f(x) = \log p(x) - \log q(x)$")

ax.plot(X_q, test_model(X_q).mean().numpy().T,
        label="posterior mean")
fill_between_stddev(X_q.squeeze(),
                    test_model(X_q).mean().numpy().squeeze(),
                    test_model(X_q).stddev().numpy().squeeze(), alpha=0.1,
                    label="posterior std dev", ax=ax)

ax.scatter(inducing_index_points, np.full_like(inducing_index_points, y_min),
           marker='^', c="tab:gray", label="inducing inputs", alpha=0.4)
ax.scatter(inducing_index_points, variational_inducing_observations_loc,
           marker='+', c="tab:blue", label="inducing variable mean")

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')

ax.legend()

plt.show()

# %%
# Density ratio.

d = tfd.Independent(tfd.LogNormal(loc=test_model(X_q).mean(),
                                  scale=test_model(X_q).stddev()),
                    reinterpreted_batch_ndims=1)

fig, ax = plt.subplots()

ax.plot(X_q, density_ratio(X_q), c='k', label=r"$r(x) = \exp f(x)$")
ax.plot(X_q, d.mean().numpy().T, label="transformed posterior mean")
fill_between_stddev(X_q.squeeze(),
                    d.mean().numpy().squeeze(),
                    d.stddev().numpy().squeeze(), alpha=0.1,
                    label="transformed posterior std dev", ax=ax)

ax.set_xlabel('$x$')
ax.set_ylabel('$r(x)$')

ax.legend()

plt.show()

# %%
# Predictive mean samples.


posterior_predictive = tf.keras.Sequential([
    test_model,
    tfp.layers.IndependentBernoulli(event_shape=(num_index_points,))
])

# %%

fig, ax = plt.subplots()

ax.plot(X_q, posterior_predictive(X_q).mean())
ax.plot(X_q, optimal_score(X_q), c='k', label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X_train, y_train, c=y_train, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()

# %%


def make_posterior_predictive(num_samples=None):

    def posterior_predictive(x):

        f_samples = test_model(x).sample(num_samples)

        return make_binary_classification_likelihood(f=f_samples)

    return posterior_predictive


posterior_predictive = make_posterior_predictive(num_samples)

# %%

fig, ax = plt.subplots()

ax.plot(X_q, posterior_predictive(X_q).mean().numpy().T, color="tab:blue",
        linewidth=0.4, alpha=0.6)
ax.plot(X_q, optimal_score(X_q), c='k', label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X_train, y_train, c=y_train, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()

# %%

y_scores = posterior_predictive(X_test).mean().numpy()
y_score_min, y_score_max = np.percentile(y_scores, q=[5, 95], axis=0)

# %%

fig, ax = plt.subplots()

ax.plot(X_q, optimal_score(X_q), c='k', label=r"$\pi(x) = \sigma(f(x))$")

ax.scatter(X_test, np.median(y_scores, axis=0), c="tab:blue", label="median",
           alpha=0.8)
ax.vlines(X_test, ymin=y_score_min, ymax=y_score_max, color="tab:blue",
          label="90\% confidence interval", alpha=0.8)

ax.scatter(X_test, y_test, c=y_test, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")

ax.set_ylabel(r"$\pi(x)$")
ax.set_xlabel(r"$x$")
ax.set_xlim(x_min, x_max)

ax.legend()

plt.show()

# %%


def get_inducing_index_points_data(inducing_index_points):

    df = pd.DataFrame(np.hstack(inducing_index_points).T)
    df.index.name = "epoch"
    df.columns.name = "inducing index points"

    s = df.stack()
    s.name = 'x'

    return s.reset_index()

# %%


data = get_inducing_index_points_data(inducing_index_points_history)

# %%


fig, ax = plt.subplots()

sns.lineplot(x='x', y="epoch", hue="inducing index points", palette="viridis",
             sort=False, data=data, alpha=0.8, ax=ax)

ax.set_xlabel(r'$x$')

plt.show()

# %%

variational_inducing_observations_scale_history = (
    history.pop("variational_inducing_observations_scale"))

# %%

fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)

im2 = ax2.imshow(variational_inducing_observations_scale_history[-1])

vmin, vmax = im2.get_clim()
im1 = ax1.imshow(variational_inducing_observations_scale_history[0],
                 vmin=vmin, vmax=vmax)

fig.colorbar(im2, ax=[ax1, ax2],
             orientation="horizontal")

ax1.set_xlabel(r"$i$")
ax1.set_ylabel(r"$j$")

ax2.set_xlabel(r"$i$")

plt.show()

# %%

history_df = pd.DataFrame(history)
history_df.index.name = "epoch"
history_df.reset_index(inplace=True)

# %%

fig, ax = plt.subplots()

sns.lineplot(x="epoch", y="nelbo", data=history_df, alpha=0.8, ax=ax)
ax.set_yscale("log")

plt.show()

# %%

parameters_df = history_df.drop(columns="nelbo") \
                          .rename(columns=lambda s: s.replace('_', ' '))

# %%

g = sns.PairGrid(parameters_df, hue="epoch", palette="RdYlBu", corner=True)
g = g.map_lower(plt.scatter, facecolor="none", alpha=0.6)
