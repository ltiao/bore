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

from utils import GOLDEN_RATIO, WIDTH, size

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

kernel_cls = kernels.ExponentiatedQuadratic

OUTPUT_DIR = "logs/figures/"


def make_test_metric(X_train, y_train, X_test, y_test):

    @np.vectorize
    def test_metric(gamma):

        # model = make_pipeline(StandardScaler(), SVR(gamma=gamma)).fit(X_train, y_train)
        model = SVR(gamma=gamma).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return mean_squared_error(y_test, y_pred)

    return test_metric


def latent(x):
    """
    Forrester's.
    """
    # return (6.0*gamma-2.0)**2 * np.sin(12.0 * gamma - 4.0)
    return np.sin(3.0*x) + x**2 - 0.7*x


def mixture(p, q, pi=0.):
    return pi*p + (1 - pi)*q


def relative(ratio_inverse, gamma):

    denom = gamma + (1-gamma) * ratio_inverse
    return 1/denom


@click.command()
@click.argument("name")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, width, aspect, extension, output_dir):

    num_features = 1
    num_init_random = 27
    noise_variance = 0.2
    gamma = 1/3
    bandwidth = 0.25

    num_index_points = 512
    x_min, x_max = -1.0, 2.0

    seed = 8888  # set random seed for reproducibility
    random_state = np.random.RandomState(seed)

    # preamble
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

    X = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)
    y = latent(X)
    # %%

    fig, ax = plt.subplots()

    ax.plot(X, y, c="tab:gray")

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$y$ (test mse)")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"objective_{suffix}.{ext}"),
                    bbox_inches="tight")

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

    fig, ax = plt.subplots()

    ax.plot(X, y)

    # ax.scatter(log_gamma_samples, y_samples, c=mask_lesser,
    #            alpha=0.7, cmap="coolwarm")
    ax.scatter(X_samples_l, y_samples_l, alpha=0.8)
    ax.scatter(X_samples_g, y_samples_g, alpha=0.8)

    ax.axhline(tau, xmin=0., xmax=1.,
               color='k', linewidth=1., linestyle='dashed')

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$y$ (test mse)")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"candidates_{suffix}.{ext}"),
                    bbox_inches="tight")

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

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ecdf_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    fig, ax = plt.subplots()

    sns.kdeplot(x=X_samples_l.squeeze(), fill=True, bw_method=bandwidth,
                label=r'$\ell(x)$', ax=ax)
    sns.kdeplot(x=X_samples_g.squeeze(), fill=True, bw_method=bandwidth,
                label=r'$g(x)$', ax=ax)
    sns.rugplot(x=X_samples_l.squeeze(), ax=ax)
    sns.rugplot(x=X_samples_g.squeeze(), ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel("density")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"kde_seaborn_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()
    # %%

    kde_l = sm.nonparametric.KDEUnivariate(X_samples_l.squeeze())
    # kde_lesser.fit(bw=BANDWIDTH)
    kde_l.fit(bw="normal_reference")

    kde_g = sm.nonparametric.KDEUnivariate(X_samples_g.squeeze())
    # kde_greater.fit(bw=BANDWIDTH)
    kde_g.fit(bw="normal_reference")

    # %%

    fig, ax = plt.subplots()

    ax.plot(X, kde_l.evaluate(X.squeeze()),
            label=fr"$\ell(x)$ -- bw {kde_l.bw:.2f}")
    ax.plot(X, kde_g.evaluate(X.squeeze()),
            label=fr"$g(x)$ -- bw {kde_g.bw:.2f}")
    sns.rugplot(x=X_samples_l.squeeze(), ax=ax)
    sns.rugplot(x=X_samples_g.squeeze(), ax=ax)

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

    ax.plot(X, kde_l.evaluate(X.squeeze()), linestyle="dashed",
            alpha=0.4, label=fr"$\ell(x)$ -- bw {kde_l.bw:.2f}")
    ax.plot(X, kde_g.evaluate(X.squeeze()), linestyle="dashed",
            alpha=0.4, label=fr"$g(x)$ -- bw {kde_g.bw:.2f}")
    ax.plot(X, kde_l.evaluate(X.squeeze()) / kde_g.evaluate(X.squeeze()),
            label=r'$\ell(x) / g(x)$')
    ax.plot(X, relative(kde_g.evaluate(X.squeeze()) / kde_l.evaluate(X.squeeze()), gamma),
            label=r'$r_{\gamma}(x)$')
    sns.rugplot(x=X_samples_l.squeeze(), ax=ax)
    sns.rugplot(x=X_samples_g.squeeze(), ax=ax)

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

    ax_main.plot(X, y, color="tab:gray", label="latent function")
    ax_main.scatter(X_samples_l, y_samples[mask_l],
                    alpha=0.8, label=r'$y < \tau$')
    ax_main.scatter(X_samples_g, y_samples[mask_g],
                    alpha=0.8, label=r'$y \geq \tau$')
    ax_main.axhline(tau, xmin=0, xmax=1.0,
                    color='gray', linewidth=1.0, linestyle='dashed')
    ax_main.annotate(rf"$\tau={{{tau:.2f}}}$", xy=(X[-1], tau),
                     xycoords='data', xytext=(-25.0, -8.0), textcoords='offset points',
                     fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    # ax_main.set_xscale("log")
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")

    ax_main.legend(loc="upper left")

    ax_x = divider.append_axes("top", size=0.9, pad=0.1, sharex=ax_main)

    ax_x.plot(X, kde_l.evaluate(X.squeeze()), label=r"$\ell(x)$")
    ax_x.plot(X, kde_g.evaluate(X.squeeze()), label=r"$g(x)$")

    sns.rugplot(x=X_samples_l.squeeze(), ax=ax_x)
    sns.rugplot(x=X_samples_g.squeeze(), ax=ax_x)

    ax_x.set_ylabel("density")
    ax_x.xaxis.set_tick_params(labelbottom=False)

    ax_x.legend()

    ax_y = divider.append_axes("right", size=0.9, pad=0.1, sharey=ax_main)

    sns.ecdfplot(y=y_samples, ax=ax_y)

    ax_y.hlines(tau, xmin=0, xmax=gamma,
                colors='gray', linestyles='dashed', linewidth=1.0)
    ax_y.vlines(gamma, ymin=y_samples.min(), ymax=tau,
                colors='gray', linestyles='dashed', linewidth=1.0)
    ax_y.annotate(rf"$\gamma={{{gamma:.2f}}}$", xy=(gamma, y_samples.min()),
                  xycoords='data', xytext=(5.0, 0.0), textcoords='offset points',
                  fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    ax_y.set_xlabel(r'$\Phi(y)$')
    ax_y.yaxis.set_tick_params(labelleft=False)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"summary_{suffix}.{ext}"),
                    bbox_inches="tight")
    plt.show()
    # %%

    kde_l = KernelDensity(kernel='gaussian', bandwidth=bandwidth) \
        .fit(X_samples_l)
    log_density_l = kde_l.score_samples(X)

    kde_g = KernelDensity(kernel='gaussian', bandwidth=bandwidth) \
        .fit(X_samples_g)
    log_density_g = kde_g.score_samples(X)
    # %%

    fig, ax = plt.subplots()

    ax.plot(X, np.exp(log_density_l), label=r'$\ell(x)$')
    ax.plot(X, np.exp(log_density_g), label=r'$g(x)$')

    sns.rugplot(x=X_samples_l.squeeze(), ax=ax)
    sns.rugplot(x=X_samples_g.squeeze(), ax=ax)

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

    ax.plot(X, np.exp(log_density_l), linestyle="dashed",
            alpha=0.4, label=fr"$\ell(x)$ -- bw {bandwidth:.2f}")
    ax.plot(X, np.exp(log_density_g), linestyle="dashed",
            alpha=0.4, label=fr"$g(x)$ -- bw {bandwidth:.2f}")
    ax.plot(X, np.exp(log_density_l - log_density_g),
            label=r'$\ell(x) / g(x)$')

    sns.rugplot(x=X_samples_l.squeeze(), ax=ax)
    sns.rugplot(x=X_samples_g.squeeze(), ax=ax)

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
        kernel=kernel, index_points=X_samples,
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
        kernel=kernel, index_points=X,
        observation_index_points=X_samples, observations=y_samples,
        observation_noise_variance=observation_noise_variance, jitter=1e-6)

    fig, ax = plt.subplots()

    ax.plot(X, gprm.mean(), label="posterior predictive mean")
    fill_between_stddev(X.squeeze(),
                        gprm.mean().numpy().squeeze(),
                        gprm.stddev().numpy().squeeze(), alpha=0.1,
                        label="posterior predictive std dev", ax=ax)

    ax.plot(X, y, label="true", color="tab:gray")
    ax.scatter(X_samples, y_samples, alpha=0.8)

    ax.legend()

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    for ext in extension:
        fig.savefig(output_path.joinpath(f"gp_posterior_predictive_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    gprm_marginals = tfd.Normal(loc=gprm.mean(), scale=gprm.stddev())

    ei = np.maximum(tau - gprm.mean(), 0.) * gprm_marginals.cdf(tau)
    ei += gprm.stddev() * gprm_marginals.prob(tau)

    fig, ax = plt.subplots()

    ax.plot(X, ei)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ei_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    gammas = np.arange(0., 0.5, 0.15)
    y_quantiles = np.quantile(y_samples, q=gammas)
    eis = (y_quantiles.reshape(-1, 1) - gprm.mean()) * gprm_marginals.cdf(y_quantiles.reshape(-1, 1))
    eis += gprm.stddev() * gprm_marginals.prob(y_quantiles.reshape(-1, 1))

    import pandas as pd

    data = pd.DataFrame(data=eis.numpy(), index=gammas, columns=X.squeeze())
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
