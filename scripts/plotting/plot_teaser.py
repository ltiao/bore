import sys
import click

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from bore.datasets import make_regression_dataset
from bore.plotting import fill_between_stddev

from sklearn.svm import SVR
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_diabetes
from sklearn.utils import check_random_state

from mpl_toolkits.axes_grid1 import make_axes_locatable

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

kernel_cls = kernels.ExponentiatedQuadratic

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499
OUTPUT_DIR = "logs/figures/"

NUM_INIT_RANDOM = 10
NUM_INDEX_POINTS = 128
PI = 0.15
BANDWIDTH = 0.1

TEST_SIZE = 0.2
SEED = 8989


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def make_test_metric(X_train, y_train, X_test, y_test):

    @np.vectorize
    def test_metric(gamma):

        # model = make_pipeline(StandardScaler(), SVR(gamma=gamma)).fit(X_train, y_train)
        model = SVR(gamma=gamma).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return mean_squared_error(y_test, y_pred)

    return test_metric


def latent(gamma):
    """
    Forrester's.
    """
    # return (6.0*gamma-2.0)**2 * np.sin(12.0 * gamma - 4.0)
    return np.sin(3.0*gamma) + gamma**2 - 0.7*gamma


def mixture(p, q, pi=0.):
    return pi*p + (1 - pi)*q


@click.command()
@click.argument("name")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, width, aspect, extension, output_dir):

    # preamble
    random_state = np.random.RandomState(SEED)

    figsize = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }

    sns.set(context="paper",
            style="ticks",
            palette="colorblind",
            font="serif",
            rc=rc)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)
    # /preamble

    # dataset = load_boston()
    # X_train, X_test, y_train, y_test = train_test_split(dataset.data,
    #                                                     dataset.target,
    #                                                     test_size=TEST_SIZE,
    #                                                     random_state=random_state)
    # test_metric = make_test_metric(X_train, y_train, X_test, y_test)
    # %%

    # log_gamma_min, log_gamma_max = -8.0, 0.0

    # # equivalent to:
    # # gamma = np.logspace(log_gamma_min, log_gamma_max, NUM_INDEX_POINTS)

    # log_gamma = np.linspace(log_gamma_min, log_gamma_max, NUM_INDEX_POINTS)
    # gamma = 10.0**log_gamma

    gamma = np.linspace(-1.0, 2.0, NUM_INDEX_POINTS)
    y = latent(gamma)
    # %%

    fig, ax = plt.subplots()

    ax.plot(gamma, y)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$y$ (test mse)")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"objective_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    # log_gamma_samples = random_state.uniform(low=log_gamma_min,
    #                                          high=log_gamma_max,
    #                                          size=NUM_INIT_RANDOM)
    # gamma_samples = 10.0**log_gamma_samples

    load_observations = make_regression_dataset(latent)
    gamma_samples, y_samples = load_observations(num_samples=NUM_INIT_RANDOM,
                                                 num_features=1,
                                                 noise_variance=0.2,
                                                 x_min=-1.0, x_max=2.0,
                                                 random_state=random_state)
    # %%

    y_threshold = np.quantile(y_samples, q=PI)
    mask_lesser = (y_samples <= y_threshold)
    mask_greater = ~mask_lesser

    gamma_samples_lesser = gamma_samples[mask_lesser]
    gamma_samples_greater = gamma_samples[mask_greater]

    # log_gamma_samples_lesser = log_gamma_samples[mask_lesser]
    # log_gamma_samples_greater = log_gamma_samples[mask_greater]

    y_samples_lesser = y_samples[mask_lesser]
    y_samples_greater = y_samples[mask_greater]
    # %%

    fig, ax = plt.subplots()

    ax.plot(gamma, y)

    # ax.scatter(log_gamma_samples, y_samples, c=mask_lesser,
    #            alpha=0.7, cmap="coolwarm")
    ax.scatter(gamma_samples_lesser, y_samples_lesser, alpha=0.8)
    ax.scatter(gamma_samples_greater, y_samples_greater, alpha=0.8)

    ax.axhline(y_threshold, xmin=0, xmax=1.0,
               color='k', linewidth=1.0, linestyle='dashed')

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$y$ (test mse)")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"candidates_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    y_samples_sorted = np.sort(y_samples)
    y_samples_quantile = np.arange(NUM_INIT_RANDOM) / NUM_INIT_RANDOM
    # %%

    fig, ax = plt.subplots()

    ax.plot(y_samples_sorted, y_samples_quantile)

    ax.vlines(y_threshold, ymin=0, ymax=PI,
              colors='k', linestyles='dashed', linewidth=1.0)
    ax.hlines(PI, xmin=y_samples_sorted[0], xmax=y_threshold,
              colors='k', linestyles='dashed', linewidth=1.0)

    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$\Phi(y)$')

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ecdf_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    fig, ax = plt.subplots()

    sns.distplot(gamma_samples_lesser, hist=False, rug=True,
                 label=r'$\ell(x)$', kde_kws=dict(shade=True, bw=BANDWIDTH), ax=ax)
    sns.distplot(gamma_samples_greater, hist=False, rug=True,
                 label=r'$g(x)$', kde_kws=dict(shade=True, bw=BANDWIDTH), ax=ax)

    ax.set_xlabel(r'$\log_{10}{x}$')
    ax.set_ylabel("density")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"kde_seaborn_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    kde_lesser = sm.nonparametric.KDEUnivariate(gamma_samples_lesser)
    # kde_lesser.fit(bw=BANDWIDTH)
    kde_lesser.fit(bw="normal_reference")

    kde_greater = sm.nonparametric.KDEUnivariate(gamma_samples_greater)
    # kde_greater.fit(bw=BANDWIDTH)
    kde_greater.fit(bw="normal_reference")

    # %%

    fig, ax = plt.subplots()

    ax.plot(gamma, kde_lesser.evaluate(gamma),
            label=fr"$\ell(x)$ -- bw {kde_lesser.bw:.2f}")
    ax.plot(gamma, kde_greater.evaluate(gamma),
            label=fr"$g(x)$ -- bw {kde_greater.bw:.2f}")

    sns.rugplot(gamma_samples_lesser, c='tab:blue', ax=ax)
    sns.rugplot(gamma_samples_greater, c='tab:orange', ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel("density")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"kde_statsmodel_normal_reference_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    fig, ax = plt.subplots()

    ax.plot(gamma, kde_lesser.evaluate(gamma), linestyle="dashed",
            alpha=0.4, label=fr"$\ell(x)$ -- bw {kde_lesser.bw:.2f}")
    ax.plot(gamma, kde_greater.evaluate(gamma), linestyle="dashed",
            alpha=0.4, label=fr"$g(x)$ -- bw {kde_greater.bw:.2f}")
    ax.plot(gamma, kde_lesser.evaluate(gamma) / kde_greater.evaluate(gamma),
            label=r'$\ell(x) / g(x)$')
    ax.plot(gamma, kde_lesser.evaluate(gamma) / mixture(kde_lesser.evaluate(gamma), kde_greater.evaluate(gamma), pi=PI),
            label=r'$r_{\gamma}(x)$')

    sns.rugplot(gamma_samples_lesser, c='tab:blue', ax=ax)
    sns.rugplot(gamma_samples_greater, c='tab:orange', ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$\alpha(x)$")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_kde_statsmodel_normal_reference_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    fig, ax_main = plt.subplots()

    divider = make_axes_locatable(ax_main)

    ax_main.plot(gamma, y, color="tab:gray", label="latent function")
    ax_main.scatter(gamma_samples_lesser, y_samples[mask_lesser],
                    alpha=0.8, label=r'$y < \tau$')
    ax_main.scatter(gamma_samples_greater, y_samples[mask_greater],
                    alpha=0.8, label=r'$y \geq \tau$')
    ax_main.axhline(y_threshold, xmin=0, xmax=1.0,
                    color='gray', linewidth=1.0, linestyle='dashed')
    ax_main.annotate(rf"$\tau={{{y_threshold:.2f}}}$", xy=(gamma[0], y_threshold),
                     xycoords='data', xytext=(-5.0, -8.0), textcoords='offset points',
                     fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    # ax_main.set_xscale("log")
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")

    ax_main.legend()

    ax_x = divider.append_axes("top", size=0.9, pad=0.1, sharex=ax_main)

    ax_x.plot(gamma, kde_lesser.evaluate(gamma), label=r"$\ell(x)$")
    ax_x.plot(gamma, kde_greater.evaluate(gamma), label=r"$g(x)$")

    sns.rugplot(gamma_samples_lesser, c='tab:blue', ax=ax_x)
    sns.rugplot(gamma_samples_greater, c='tab:orange', ax=ax_x)

    ax_x.set_ylabel("density")
    ax_x.xaxis.set_tick_params(labelbottom=False)

    ax_x.legend()

    ax_y = divider.append_axes("right", size=0.9, pad=0.1, sharey=ax_main)

    ax_y.plot(y_samples_quantile, y_samples_sorted)
    ax_y.hlines(y_threshold, xmin=0, xmax=PI,
                colors='gray', linestyles='dashed', linewidth=1.0)
    ax_y.vlines(PI, ymin=y_samples_sorted[0], ymax=y_threshold,
                colors='gray', linestyles='dashed', linewidth=1.0)
    ax_y.annotate(rf"$\gamma={{{PI:.2f}}}$", xy=(PI, y_samples_sorted[0]),
                  xycoords='data', xytext=(5.0, 0.0), textcoords='offset points',
                  fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    ax_y.set_xlabel(r'$\Phi(y)$')
    ax_y.yaxis.set_tick_params(labelleft=False)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"summary_{suffix}.{ext}"),
                    bbox_inches="tight")
    plt.show()
    # %%

    kde_lesser = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH) \
        .fit(gamma_samples_lesser.reshape(-1, 1))
    log_density_lesser = kde_lesser.score_samples(gamma.reshape(-1, 1))

    kde_greater = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH) \
        .fit(gamma_samples_greater.reshape(-1, 1))
    log_density_greater = kde_greater.score_samples(gamma.reshape(-1, 1))
    # %%

    fig, ax = plt.subplots()

    ax.plot(gamma, np.exp(log_density_lesser), label=r'$\ell(x)$')
    ax.plot(gamma, np.exp(log_density_greater), label=r'$g(x)$')

    sns.rugplot(gamma_samples_lesser, c='tab:blue', ax=ax)
    sns.rugplot(gamma_samples_greater, c='tab:orange', ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel("density")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"kde_sklearn_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    fig, ax = plt.subplots()

    ax.plot(gamma, np.exp(log_density_lesser), linestyle="dashed",
            alpha=0.4, label=fr"$\ell(x)$ -- bw {BANDWIDTH:.2f}")
    ax.plot(gamma, np.exp(log_density_greater), linestyle="dashed",
            alpha=0.4, label=fr"$g(x)$ -- bw {BANDWIDTH:.2f}")
    ax.plot(gamma, np.exp(log_density_lesser - log_density_greater),
            label=r'$\ell(x) / g(x)$')

    sns.rugplot(gamma_samples_lesser, c='tab:blue', ax=ax)
    sns.rugplot(gamma_samples_greater, c='tab:orange', ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$\alpha(x)$")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_kde_sklearn_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    amplitude = tfp.util.TransformedVariable(
        1.0, bijector=tfp.bijectors.Softplus(), dtype="float64", name='amplitude')
    length_scale = tfp.util.TransformedVariable(
        0.5, bijector=tfp.bijectors.Softplus(), dtype="float64", name='length_scale')
    observation_noise_variance = tfp.util.TransformedVariable(
        1e-1, bijector=tfp.bijectors.Softplus(), dtype="float64",
        name='observation_noise_variance')

    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)
    gp = tfd.GaussianProcess(
        kernel=kernel, index_points=gamma_samples.reshape(-1, 1),
        observation_noise_variance=observation_noise_variance)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.5,
                                         beta_2=0.99)

    num_epochs = 200

    for epoch in range(num_epochs):

        with tf.GradientTape() as tape:
            nll = - gp.log_prob(y_samples)

        gradients = tape.gradient(nll, gp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gp.trainable_variables))

    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel, index_points=gamma.reshape(-1, 1),
        observation_index_points=gamma_samples.reshape(-1, 1), observations=y_samples,
        observation_noise_variance=observation_noise_variance, jitter=1e-6)

    fig, ax = plt.subplots()

    ax.plot(gamma, gprm.mean(), label="posterior predictive mean")
    fill_between_stddev(gamma,
                        gprm.mean().numpy().squeeze(),
                        gprm.stddev().numpy().squeeze(), alpha=0.1,
                        label="posterior predictive std dev", ax=ax)

    ax.plot(gamma, y, label="true", color="tab:gray")
    ax.scatter(gamma_samples, y_samples, alpha=0.8)

    ax.legend()

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    for ext in extension:
        fig.savefig(output_path.joinpath(f"gp_posterior_predictive_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    gprm_marginals = tfd.Normal(loc=gprm.mean(), scale=gprm.stddev())

    ei = np.maximum(y_threshold - gprm.mean(), 0.) * gprm_marginals.cdf(y_threshold)
    ei += gprm.stddev() * gprm_marginals.prob(y_threshold)

    fig, ax = plt.subplots()

    ax.plot(gamma, ei)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ei_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    pis = np.arange(0., 0.5, 0.15)
    y_quantiles = np.quantile(y_samples, q=pis)
    eis = (y_quantiles.reshape(-1, 1) - gprm.mean()) * gprm_marginals.cdf(y_quantiles.reshape(-1, 1))
    eis += gprm.stddev() * gprm_marginals.prob(y_quantiles.reshape(-1, 1))

    import pandas as pd

    data = pd.DataFrame(data=eis.numpy(), index=pis, columns=gamma)
    data.index.name = r"$\gamma$"
    data.columns.name = r"$x$"
    s = data.stack()
    s.name = "ei"
    df = s.reset_index()

    fig, ax = plt.subplots()

    sns.lineplot(x=r"$x$", y="ei", hue=r"$\gamma$", data=df, ax=ax)
    # ax.plot(gamma, eis.numpy().T)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"eis_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
