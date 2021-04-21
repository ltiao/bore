import sys
import click

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

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


from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import GOLDEN_RATIO, WIDTH, pt_to_in

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

kernel_cls = kernels.ExponentiatedQuadratic

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
@click.option('--gamma', '-g', type=float, default=1/3)
@click.option("--output-dir", default="figures/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="deep")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--seed", default=8888)
def main(name, gamma, output_dir, transparent, context, style, palette, width,
         height, aspect, dpi, extension, seed):

    num_features = 1
    num_init_random = 27
    noise_variance = 0.2
    bandwidth = 0.25

    num_index_points = 512
    x_min, x_max = -1.0, 2.0

    random_state = np.random.RandomState(seed)

    # preamble
    if height is None:
        height = width / aspect
    # figsize = size(width, aspect)
    figsize = (width, height)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)
    # / preamble

    X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)
    y_grid = latent(X_grid)

    fig, ax = plt.subplots()

    ax.plot(X_grid, y_grid, c="tab:gray")

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$y$ (test mse)")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"objective_{suffix}.{ext}"), dpi=dpi,
                    transparent=transparent)

    plt.show()
    # %%

    load_observations = make_regression_dataset(latent)
    X, y = load_observations(num_samples=num_init_random,
                             num_features=num_features,
                             noise_variance=noise_variance,
                             x_min=x_min, x_max=x_max,
                             random_state=random_state)
    # %%
    tau = np.quantile(y, q=gamma, interpolation="lower")
    z = np.less_equal(y, tau)

    Xl = X[z]
    Xg = X[~z]

    yl = y[z]
    yg = y[~z]

    # %%
    fig, ax = plt.subplots()

    ax.plot(X_grid, y_grid, color="tab:gray", label="latent function")
    ax.scatter(X, y, marker='x', color='k',
               label="noisy observations")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    ax.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"observations_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()
    # %%

    fig, ax = plt.subplots()

    sns.ecdfplot(x=y, ax=ax)

    ax.axvline(tau, ymin=0., ymax=gamma,
               color="black", linestyle='dashed', linewidth=1.0)
    ax.hlines(gamma, xmin=y.min(), xmax=tau,
              colors="black", linestyles='dashed', linewidth=1.0)

    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$\Phi(y)$')

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ecdf_{suffix}.{ext}"), dpi=dpi,
                    transparent=transparent)

    plt.show()
    # %%

    fig, ax = plt.subplots()

    divider = make_axes_locatable(ax)

    ax.plot(X_grid, y_grid, color="tab:gray", label="latent function")
    ax.scatter(X, y, marker='x', color='k',
               label="noisy observations")

    ax.axhline(y.min(), xmin=0.0, xmax=1.0, color='k', linewidth=1.0, linestyle='dashed')
    ax.annotate(rf"$\tau=\min_n y_n$", xy=(X_grid[-1], y.min()),
                xycoords='data', xytext=(-35.0, 4.0), textcoords='offset points',
                fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    ax.axhline(tau, xmin=0.0, xmax=1.0, color='k', linewidth=1.0, linestyle='dashed')
    ax.annotate(rf"$\tau={{{tau:.2f}}}$", xy=(X_grid[-1], tau),
                xycoords='data', xytext=(-35.0, 4.0), textcoords='offset points',
                fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    ax.legend(loc="upper left")

    ax_y = divider.append_axes("right", size=0.9, pad=0.1, sharey=ax)

    sns.ecdfplot(y=y, c='tab:gray', ax=ax_y)

    ax_y.hlines(tau, xmin=0, xmax=gamma,
                colors='k', linestyles='dashed', linewidth=1.0)
    ax_y.vlines(gamma, ymin=y.min(), ymax=tau,
                colors='k', linestyles='dashed', linewidth=1.0)
    ax_y.annotate(rf"$\gamma={{{gamma:.2f}}}$", xy=(gamma, y.min()),
                  xycoords='data', xytext=(5.0, 0.0), textcoords='offset points',
                  fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    ax_y.set_xlabel(r"$\Phi(y)$")
    ax_y.yaxis.set_tick_params(labelleft=False)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"observations_ecdf_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()
    # %%

    fig, ax = plt.subplots()

    sns.kdeplot(x=Xl.squeeze(), fill=True, bw_method=bandwidth,
                label=r'$\ell(x)$', ax=ax)
    sns.kdeplot(x=Xg.squeeze(), fill=True, bw_method=bandwidth,
                label=r'$g(x)$', ax=ax)
    sns.rugplot(x=Xl.squeeze(), ax=ax)
    sns.rugplot(x=Xg.squeeze(), ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel("density")

    ax.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"kde_seaborn_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()
    # %%

    kde_l = sm.nonparametric.KDEUnivariate(Xl.squeeze())
    # kde_lesser.fit(bw=BANDWIDTH)
    kde_l.fit(bw="normal_reference")

    kde_g = sm.nonparametric.KDEUnivariate(Xg.squeeze())
    # kde_greater.fit(bw=BANDWIDTH)
    kde_g.fit(bw="normal_reference")

    # %%
    fig, ax = plt.subplots()

    ax.plot(X_grid, kde_l.evaluate(X_grid.squeeze()),
            label=fr"$\ell(x)$ -- bw {kde_l.bw:.2f}")
    ax.plot(X_grid, kde_g.evaluate(X_grid.squeeze()),
            label=fr"$g(x)$ -- bw {kde_g.bw:.2f}")
    sns.rugplot(x=Xl.squeeze(), ax=ax)
    sns.rugplot(x=Xg.squeeze(), ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel("density")

    ax.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"kde_statsmodel_normal_reference_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()
    # %%

    fig, ax = plt.subplots()

    ax.plot(X_grid, kde_l.evaluate(X_grid.squeeze()), linestyle="dashed",
            alpha=0.4, label=fr"$\ell(x)$ -- bw {kde_l.bw:.2f}")
    ax.plot(X_grid, kde_g.evaluate(X_grid.squeeze()), linestyle="dashed",
            alpha=0.4, label=fr"$g(x)$ -- bw {kde_g.bw:.2f}")
    ax.plot(X_grid, kde_l.evaluate(X_grid.squeeze()) / kde_g.evaluate(X_grid.squeeze()),
            label=r'$\ell(x) / g(x)$')
    ax.plot(X_grid, relative(kde_g.evaluate(X_grid.squeeze()) / kde_l.evaluate(X_grid.squeeze()), gamma),
            label=r'$r_{\gamma}(x)$')
    sns.rugplot(x=Xl.squeeze(), ax=ax)
    sns.rugplot(x=Xg.squeeze(), ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$\alpha(x)$")

    ax.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_kde_statsmodel_normal_reference_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()
    # %%
    # START HERE

    fig, ax = plt.subplots()

    # ax.set_title(f"Iteration {i+1:d}")

    ax.plot(X_grid, y_grid, color="tab:gray", label="latent function",
            zorder=-1)

    ax.scatter(X[z], y[z], marker='x', alpha=0.9,
               label=r'observations $y < \tau$', zorder=2)
    ax.scatter(X[~z], y[~z], marker='x', alpha=0.9,
               label=r'observations $y \geq \tau$', zorder=2)

    ax.axhline(tau, xmin=0.0, xmax=1.0, color='k',
               linewidth=1.0, linestyle='dashed', zorder=5)

    ax.annotate(rf"$\tau={{{tau:.2f}}}$", xy=(X_grid.max(), tau),
                xycoords='data', xytext=(10, 2), textcoords='offset points',
                # arrowprops=dict(facecolor='black', shrink=0.05),
                # bbox=dict(boxstyle="round"),
                fontsize="xx-small", horizontalalignment='right', verticalalignment='bottom')

    # if i < num_iterations - 1:
    #     ax.axvline(x_next, ymin=0.0, ymax=1.0, color=next_loc_color,
    #                alpha=0.8, linewidth=1.0, label="next location",
    #                zorder=10)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend(loc="upper left")

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=0.6, pad=0.1, sharex=ax)

    # ax_top.plot(X_grid, kde_l.evaluate(X_grid.squeeze()), label=r"$\ell(x)$")
    # ax_top.plot(X_grid, kde_g.evaluate(X_grid.squeeze()), label=r"$g(x)$")

    sns.kdeplot(x=Xl.squeeze(), fill=True, clip=(1.1*X_grid.min(), 1.1*X_grid.max()),
                bw_method=0.25,  # "silverman",
                label=r'$\ell(x)$', ax=ax_top)
    sns.kdeplot(x=Xg.squeeze(), fill=True, clip=(1.1*X_grid.min(), 1.1*X_grid.max()),
                bw_method=0.2,  # "silverman",
                label=r'$g(x)$', ax=ax_top)

    ax_top.set_prop_cycle(None)

    sns.rugplot(x=Xl.squeeze(), height=0.1, ax=ax_top)
    sns.rugplot(x=Xg.squeeze(), height=0.1, ax=ax_top)

    ax_top.set_xlim(1.1*X_grid.min(), 1.1*X_grid.max())

    ax_top.legend(loc="upper left")

    ax_top.set_ylabel("density")
    ax_top.xaxis.set_tick_params(labelbottom=False)

    # ax_top.scatter(X[z], np.ones_like(X[z]), marker='s',
    #                edgecolors="none", alpha=0.7, zorder=2)
    # ax_top.scatter(X[~z], np.zeros_like(X[~z]), marker='s',
    #                edgecolors="none", alpha=0.7, zorder=2)

    # ax_top.plot(X_grid, tf.sigmoid(model.predict(X_grid)), c='tab:gray',
    #             label=r"$\pi_{\theta}(x)$", zorder=-1)

    # if i < num_iterations - 1:
    #     ax_top.axvline(x_next, ymin=0.0, ymax=1.0, color=next_loc_color,
    #                    alpha=0.8, linewidth=1.0, zorder=10)

    ax_right = divider.append_axes("right", size=0.6, pad=0.1, sharey=ax)

    sns.ecdfplot(y=y, c='tab:gray', ax=ax_right, zorder=-1)

    ax_right.scatter(gamma, tau, c='k', marker='.', zorder=5)
    ax_right.hlines(tau, xmin=0, xmax=gamma, colors='k', linestyles='dashed', linewidth=1.0, zorder=5)
    ax_right.vlines(gamma, ymin=1.25*y_grid.min(), ymax=tau, colors='k', linestyles='dashed', linewidth=1.0, zorder=5)

    ax_right.annotate(rf"$\gamma={{{gamma:.2f}}}$",
                      xy=(gamma, 1.25*y_grid.min()),
                      xycoords='data', xytext=(-2, -3),
                      textcoords='offset points',
                      fontsize="xx-small",
                      # arrowprops=dict(facecolor='black', shrink=0.05),
                      # bbox=dict(boxstyle="round", fc="none"),
                      horizontalalignment='left', verticalalignment='top')

    ax_right.set_xlabel(r'$\Phi(y)$')
    ax_right.yaxis.set_tick_params(labelleft=False)

    ax_right.set_ylim(1.25*y_grid.min(), 1.05*y_grid.max())

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"header_{suffix}.{ext}"), dpi=dpi, transparent=transparent)

    plt.show()

    # HERE

    fig, ax_main = plt.subplots()

    divider = make_axes_locatable(ax_main)

    ax_main.plot(X_grid, y_grid, color="tab:gray", label="latent function")
    ax_main.scatter(Xl, yl,
                    marker='x', alpha=0.8, label=r'observations $y < \tau$')
    ax_main.scatter(Xg, yg,
                    marker='x', alpha=0.8, label=r'observations $y \geq \tau$')
    ax_main.axhline(tau, xmin=0, xmax=1.0,
                    color='k', linewidth=1.0, linestyle='dashed')
    ax_main.annotate(rf"$\tau={{{tau:.2f}}}$", xy=(X_grid[-1], tau),
                     xycoords='data', xytext=(-30.0, -8.0), textcoords='offset points',
                     fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    # ax_main.set_xscale("log")
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")

    ax_main.legend(loc="upper left")

    ax_x = divider.append_axes("top", size=0.8, pad=0.1, sharex=ax_main)

    ax_x.plot(X_grid, kde_l.evaluate(X_grid.squeeze()), label=r"$\ell(x)$")
    ax_x.plot(X_grid, kde_g.evaluate(X_grid.squeeze()), label=r"$g(x)$")

    ax_x.set_prop_cycle(None)

    sns.rugplot(x=Xl.squeeze(), height=0.1, ax=ax_x)
    sns.rugplot(x=Xg.squeeze(), height=0.1, ax=ax_x)

    ax_x.set_ylabel("density")
    ax_x.xaxis.set_tick_params(labelbottom=False)

    ax_x.legend(loc="upper left")

    ax_y = divider.append_axes("right", size=0.9, pad=0.1, sharey=ax_main)

    sns.ecdfplot(y=y, c='tab:gray', ax=ax_y)

    ax_y.hlines(tau, xmin=0, xmax=gamma,
                colors='k', linestyles='dashed', linewidth=1.0)
    ax_y.vlines(gamma, ymin=y.min(), ymax=tau,
                colors='k', linestyles='dashed', linewidth=1.0)
    ax_y.annotate(rf"$\gamma={{{gamma:.2f}}}$", xy=(gamma, y.min()),
                  xycoords='data', xytext=(5.0, 0.0), textcoords='offset points',
                  fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

    ax_y.set_xlabel(r'$\Phi(y)$')
    ax_y.yaxis.set_tick_params(labelleft=False)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"summary_{suffix}.{ext}"), dpi=dpi, transparent=transparent)

    plt.show()
    # %%

    kde_l = KernelDensity(kernel='gaussian', bandwidth=bandwidth) \
        .fit(Xl)
    log_density_l = kde_l.score_samples(X_grid)

    kde_g = KernelDensity(kernel='gaussian', bandwidth=bandwidth) \
        .fit(Xg)
    log_density_g = kde_g.score_samples(X_grid)
    # %%

    fig, ax = plt.subplots()

    ax.plot(X_grid, np.exp(log_density_l), label=r'$\ell(x)$')
    ax.plot(X_grid, np.exp(log_density_g), label=r'$g(x)$')

    sns.rugplot(x=Xl.squeeze(), ax=ax)
    sns.rugplot(x=Xg.squeeze(), ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel("density")

    ax.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"kde_sklearn_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()
    # %%

    fig, ax = plt.subplots()

    ax.plot(X_grid, np.exp(log_density_l), linestyle="dashed",
            alpha=0.4, label=fr"$\ell(x)$ -- bw {bandwidth:.2f}")
    ax.plot(X_grid, np.exp(log_density_g), linestyle="dashed",
            alpha=0.4, label=fr"$g(x)$ -- bw {bandwidth:.2f}")
    ax.plot(X_grid, np.exp(log_density_l - log_density_g),
            label=r'$\ell(x) / g(x)$')

    sns.rugplot(x=Xl.squeeze(), ax=ax)
    sns.rugplot(x=Xg.squeeze(), ax=ax)

    # ax.set_xscale("log")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$\alpha(x)$")

    ax.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_kde_sklearn_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

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
        kernel=kernel, index_points=X,
        observation_noise_variance=observation_noise_variance)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.5,
                                         beta_2=0.99)

    num_epochs = 200

    for epoch in range(num_epochs):

        with tf.GradientTape() as tape:
            nll = - gp.log_prob(y)

        gradients = tape.gradient(nll, gp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gp.trainable_variables))

    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel, index_points=X_grid,
        observation_index_points=X, observations=y,
        observation_noise_variance=observation_noise_variance, jitter=1e-6)

    fig, ax = plt.subplots()

    ax.plot(X_grid, gprm.mean(), label="posterior predictive mean")
    fill_between_stddev(X_grid.squeeze(),
                        gprm.mean().numpy().squeeze(),
                        gprm.stddev().numpy().squeeze(), alpha=0.1,
                        label="posterior predictive std dev", ax=ax)

    ax.plot(X_grid, y_grid, label="true", color="tab:gray")
    ax.scatter(X, y, marker='x', alpha=0.8)

    ax.legend()

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"gp_posterior_predictive_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    gprm_marginals = tfd.Normal(loc=gprm.mean(), scale=gprm.stddev())

    ei = np.maximum(tau - gprm.mean(), 0.) * gprm_marginals.cdf(tau)
    ei += gprm.stddev() * gprm_marginals.prob(tau)

    fig, ax = plt.subplots()

    ax.plot(X_grid, ei)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ei_{suffix}.{ext}"), dpi=dpi,
                    transparent=transparent)

    plt.show()

    gammas = np.arange(0., 0.5, 0.15)
    y_quantiles = np.quantile(y, q=gammas)

    pis = gprm_marginals.cdf(y_quantiles.reshape(-1, 1))

    eis = (y_quantiles.reshape(-1, 1) - gprm.mean()) * gprm_marginals.cdf(y_quantiles.reshape(-1, 1))
    eis += gprm.stddev() * gprm_marginals.prob(y_quantiles.reshape(-1, 1))

    frames = []
    for kind, arr in [("PI", pis), ("EI", eis)]:
        data = pd.DataFrame(data=arr.numpy(), index=gammas, columns=X_grid.squeeze())
        data.index.name = "gamma"
        data.columns.name = "x"
        s = data.stack()
        s.name = "y"
        frame = s.reset_index()
        frames.append(frame.assign(kind=kind))

    data = pd.concat(frames, axis="index", sort=True)

    fig, ax = plt.subplots()

    sns.lineplot(x="x", y="y", hue="gamma", style="kind", data=data, ax=ax)

    # ax.plot(gamma, eis.numpy().T)
    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"eis_{suffix}.{ext}"), dpi=dpi,
                    transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
