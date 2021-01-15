import sys
import click

import numpy as np
import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow_probability as tfp

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from pathlib import Path

from tensorflow.keras.losses import BinaryCrossentropy
from bore.models import DenseMaximizableSequential
from bore.datasets import make_classification_dataset
from utils import GOLDEN_RATIO, WIDTH, pt_to_in

from sklearn.svm import SVC


K.set_floatx("float64")

# shortcuts
tfd = tfp.distributions

OUTPUT_DIR = "figures/"


class DensityRatioBase(ABC):

    def __call__(self, X, y=None):

        return self.ratio(X, y)

    @abstractmethod
    def logit(self, X, y=None):
        pass

    def ratio(self, X, y=None):

        return tf.exp(self.logit(X, y))

    def prob(self, X, y=None):
        """
        Probability of sample being from P_{top}(x) vs. P_{bot}(x).
        """
        return tf.sigmoid(self.logit(X, y))


class DensityRatioMarginals(DensityRatioBase):

    def __init__(self, top, bot):

        self.top = top
        self.bot = bot

    def logit(self, X, y=None):

        return self.top.log_prob(X) - self.bot.log_prob(X)

    def make_dataset(self, num_samples, rate=0.5, dtype=tf.float64, seed=None):

        num_top = int(num_samples * rate)
        num_bot = num_samples - num_top

        _X_top = self.top.sample(sample_shape=(num_top, 1), seed=seed)
        _X_bot = self.bot.sample(sample_shape=(num_bot, 1), seed=seed)

        X_top = tf.cast(_X_top, dtype=dtype).numpy()
        X_bot = tf.cast(_X_bot, dtype=dtype).numpy()

        return X_top, X_bot


class MLPDensityRatioEstimator(DensityRatioBase):

    def __init__(self, num_layers=2, num_units=32, activation="tanh",
                 seed=None, *args, **kwargs):

        self.model = DenseMaximizableSequential(1, num_layers, num_units,
                                                layer_kws=dict(activation=activation))

    def logit(self, X, y=None):

        # TODO: time will tell whether squeezing the final axis
        # makes things easier.
        return K.squeeze(self.model(X), axis=-1)

    def compile(self, optimizer, metrics=["accuracy"], *args, **kwargs):

        self.model.compile(optimizer=optimizer,
                           loss=BinaryCrossentropy(from_logits=True),
                           metrics=metrics, *args, **kwargs)

    def fit(self, X_top, X_bot, *args, **kwargs):

        X, y = make_classification_dataset(X_top, X_bot)
        return self.model.fit(X, y, *args, **kwargs)

    def evaluate(self, X_top, X_bot, *args, **kwargs):

        X, y = make_classification_dataset(X_top, X_bot)
        return self.model.evaluate(X, y, *args, **kwargs)


def gamma_relative_density_ratio(ratio, gamma):

    denom = gamma + (1-gamma) / ratio
    return 1 / denom


@click.command()
@click.argument("name")
@click.option('--gamma', '-g', type=float, default=1/3)
@click.option('--estimation/--no-estimation', default=False)
@click.option("--output-dir", default=OUTPUT_DIR,
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
def main(name, gamma, estimation, output_dir, transparent, context, style,
         palette, width, height, aspect, dpi, extension, seed):

    num_features = 1  # dimensionality
    num_train = 1000  # nbr training points in synthetic dataset
    # x_min, x_max = -6.0, 6.0
    x_min, x_max = -5.0, 5.0
    num_index_points = 512  # nbr of index points

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

    random_state = np.random.RandomState(seed)
    # /preamble

    X_grid = np.linspace(x_min, x_max, num_index_points) \
               .reshape(-1, num_features)

    # p = tfd.MixtureSameFamily(
    #     mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
    #     components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
    # q = tfd.Normal(loc=0.0, scale=2.0)

    p = tfd.Normal(loc=0.0, scale=1.0)
    q = tfd.Normal(loc=0.5, scale=1.0)

    # p = tfd.Normal(loc=1.0, scale=1.0)
    # q = tfd.Normal(loc=0.0, scale=2.0)

    r = DensityRatioMarginals(top=p, bot=q)

    X_p, X_q = r.make_dataset(num_train, rate=gamma, seed=seed)
    X_train, y_train = make_classification_dataset(X_p, X_q)

    kde_lesser = sm.nonparametric.KDEUnivariate(X_p)
    kde_lesser.fit(bw="normal_reference")

    kde_greater = sm.nonparametric.KDEUnivariate(X_q)
    kde_greater.fit(bw="normal_reference")

    # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex="col")

    fig, ax1 = plt.subplots()

    ax1.plot(X_grid.squeeze(axis=-1),
             r.top.prob(X_grid).numpy().squeeze(axis=-1), label=r"$\ell(x)$")
    ax1.plot(X_grid.squeeze(axis=-1),
             r.bot.prob(X_grid).numpy().squeeze(axis=-1), label=r"$g(x)$")

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel('density')

    ax1.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"densities_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax2 = plt.subplots()

    ax2.plot(X_grid.squeeze(axis=-1),
             r.ratio(X_grid).numpy().squeeze(axis=-1), label=r"$r_0(x)$",
             color="tab:orange")
    ax2.plot(X_grid.squeeze(axis=-1),
             gamma_relative_density_ratio(r.ratio(X_grid), gamma=gamma)
                   .numpy().squeeze(axis=-1), label=fr"$r_{{{gamma:.2f}}}(x)$",
             color="tab:green")

    # ax2.set_ylim(-0.1, 5.0)

    # ax2.plot(X_grid.squeeze(axis=-1),
    #          r.ratio(X_grid).numpy().squeeze(axis=-1), label=r"$\frac{\ell(x)}{g(x)}$",
    #          color="tab:orange")
    # # ax2.plot(X_grid.squeeze(axis=-1),
    # #          gamma_relative_density_ratio(r.ratio(X_grid), gamma=gamma)
    # #                .numpy().squeeze(axis=-1), label=fr"$r_{{{gamma:.2f}}}(x)$",
    # #          color="tab:green")

    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel('density ratio')

    ax2.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"density_ratios_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0

    # Build DataFrame
    rows = []
    # rows.append(dict(x=X_grid.squeeze(axis=-1),
    #                  y=r.top.prob(X_grid).numpy().squeeze(axis=-1),
    #                  density=r"$\ell(x)$", kind=r"$\textsc{exact}$"))
    # rows.append(dict(x=X_grid.squeeze(axis=-1),
    #                  y=r.bot.prob(X_grid).numpy().squeeze(axis=-1),
    #                  density=r"$g(x)$", kind=r"$\textsc{exact}$"))

    rows.append(dict(x=X_grid.squeeze(axis=-1),
                     y=r.top.prob(X_grid).numpy().squeeze(axis=-1),
                     kind=r"$\ell(x)$"))
    rows.append(dict(x=X_grid.squeeze(axis=-1),
                     y=r.bot.prob(X_grid).numpy().squeeze(axis=-1),
                     kind=r"$g(x)$"))

    rows.append(dict(x=X_grid.squeeze(axis=-1),
                     y=r.ratio(X_grid).numpy().squeeze(axis=-1),
                     kind=r"$r_0(x)$"))
    rows.append(dict(x=X_grid.squeeze(axis=-1),
                     y=gamma_relative_density_ratio(r.ratio(X_grid), gamma=gamma) \
                             .numpy().squeeze(axis=-1),
                     kind=fr"$r_{{{gamma:.2f}}}(x)$"))

    if estimation:
        rows.append(dict(x=X_grid.squeeze(axis=-1),
                         y=kde_lesser.evaluate(X_grid.ravel()),
                         density=r"$\ell(x)$", kind=r"$\textsc{kde}$"))
        rows.append(dict(x=X_grid.squeeze(axis=-1),
                         y=kde_greater.evaluate(X_grid.ravel()),
                         density=r"$g(x)$", kind=r"$\textsc{kde}$"))

    frames = map(pd.DataFrame, rows)
    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    fig, ax = plt.subplots()

    sns.lineplot(x='x', y='y', hue="kind", data=data, ax=ax)

    # sns.lineplot(x='x', y='y', hue="density", style="kind", data=data, ax=ax)

    # sns.rugplot(X_p.squeeze(), height=0.02, c='tab:blue', alpha=0.2, ax=ax)
    # sns.rugplot(X_q.squeeze(), height=0.02, c='tab:orange', alpha=0.2, ax=ax)

    ax.set_xlabel('$x$')
    ax.set_ylabel('density')

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"densities_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    if not estimation:
        return 0

    # clf = SVC(C=100.0, kernel="rbf", probability=True, tol=1e-9).fit(X_train, y_train)

    r_mlp = MLPDensityRatioEstimator(num_layers=3, num_units=32, activation="elu")
    r_mlp.compile(optimizer="adam", metrics=["accuracy"])
    r_mlp.fit(X_p, X_q, epochs=500, batch_size=64)

    # Build DataFrame
    rows = []
    # exact
    rows.append({'x': X_grid.squeeze(axis=-1),
                 'y': r.ratio(X_grid).numpy().squeeze(axis=-1),
                 'kind': r"$\textsc{exact}$", r'$\gamma$': r"$0$"})
    rows.append({'x': X_grid.squeeze(axis=-1),
                 'y': gamma_relative_density_ratio(r.ratio(X_grid), gamma=gamma) \
                            .numpy().squeeze(axis=-1),
                 'kind': r"$\textsc{exact}$", r'$\gamma$': r"$\frac{1}{3}$"})
    # cpe
    rows.append({'x': X_grid.squeeze(axis=-1),
                 # 'y': np.exp(- clf.decision_function(X_grid) * clf.probA_ + clf.probB_) * (1 - gamma) / gamma,
                 'y': r_mlp.ratio(X_grid) * (1 - gamma) / gamma,
                 'kind': r"$\textsc{cpe}$", r'$\gamma$': r"$0$"})
    rows.append({'x': X_grid.squeeze(axis=-1),
                 'y': r_mlp.prob(X_grid) / gamma,
                 # 'y': clf.predict_proba(X_grid).T[1] / gamma,
                 'kind': r"$\textsc{cpe}$", r'$\gamma$': r"$\frac{1}{3}$"})
    # kde
    rows.append({'x': X_grid.squeeze(axis=-1),
                 'y': kde_lesser.evaluate(X_grid.ravel()) / kde_greater.evaluate(X_grid.ravel()),
                 'kind': r"$\textsc{kde}$", r'$\gamma$': r"$0$"})
    rows.append({'x': X_grid.squeeze(axis=-1),
                 'y': gamma_relative_density_ratio(kde_lesser.evaluate(X_grid.ravel()) / kde_greater.evaluate(X_grid.ravel()), gamma),
                 'kind': r"$\textsc{kde}$", r'$\gamma$': r"$\frac{1}{3}$"})
    data = pd.concat(map(pd.DataFrame, rows), axis="index", ignore_index=True,
                     sort=True)

    fig, ax = plt.subplots()

    sns.lineplot(x='x', y='y', hue="kind", style=r"$\gamma$", palette="Set1",
                 data=data, ax=ax)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$r_{\gamma}(x)$")

    # ax.set_ylim(-0.01, 1/gamma+0.1)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratios_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
