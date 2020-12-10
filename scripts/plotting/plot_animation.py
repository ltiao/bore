import sys
import click

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

from scipy.optimize import Bounds

from bore.benchmarks import Forrester
from bore.models import MaximizableSequential

from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import GOLDEN_RATIO, WIDTH, pt_to_in


@click.command()
@click.argument("name")
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option("--num-iterations", "-i", default=6)
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option('--seed', '-s', default=1)
def main(name, output_dir, num_iterations, transparent, context, style,
         palette, width, height, aspect, dpi, extension, seed):

    next_loc_color = "tab:green"

    num_index_points = 512
    num_random_init = 4
    num_features = 1

    gamma = 0.25

    num_units = 32
    activation = "softplus"
    optimizer = "adam"
    epochs = 500
    batch_size = 64

    noise_scale = 0.05

    random_state = np.random.RandomState(seed)

    # preamble
    if height is None:
        height = width / aspect
    # height *= num_iterations
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

    benchmark = Forrester()
    cs = benchmark.get_config_space()
    hx = cs.get_hyperparameter("x")
    bounds = Bounds(lb=[hx.lower], ub=[hx.upper])
    X_grid = np.linspace(hx.lower, hx.upper, num_index_points).reshape(-1, num_features)
    y_grid = benchmark.func(X_grid)

    model = MaximizableSequential()
    model.add(Dense(num_units, input_dim=num_features, activation=activation))
    model.add(Dense(num_units, activation=activation))
    model.add(Dense(num_units, activation=activation))
    model.add(Dense(1))
    # model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(from_logits=True))  # loss="binary_crossentropy")
    model.summary(print_fn=click.echo)

    # random initial design
    features = []
    targets = []

    # for i in range(num_random_init):

        # propose new point
        # x_new = random_state.uniform(low=hx.lower, high=hx.upper, size=(1,))

    initial_designs = [0.025, 0.15, 0.825, 0.975]
    for x_new in initial_designs:

        # evaluate
        y_new = benchmark.func(x_new) + random_state.normal(scale=noise_scale)

        features.append(x_new)
        targets.append(y_new)

    # fig, axes = plt.subplots(nrows=num_iterations, sharex=True)
    # for i, ax in enumerate(axes):

    for i in range(num_iterations):

        # construct binary classification problem
        X = np.vstack(features)
        y = np.hstack(targets)
        tau = np.quantile(y, q=gamma, interpolation="lower")
        z = np.less_equal(y, tau)

        # update classifier
        model.fit(X, z, epochs=epochs, batch_size=batch_size, verbose=False)

        # suggest new candidate
        x_new = model.argmax(bounds=bounds, print_fn=click.echo)

        # evaluate blackbox
        y_new = benchmark.func(x_new) + random_state.normal(scale=noise_scale)

        # update dataset
        features.append(x_new)
        targets.append(y_new)

        fig, ax = plt.subplots()

        # ax.set_title(f"Iteration {i+1:d}")

        ax.plot(X_grid, y_grid, color="tab:gray", label="latent function",
                zorder=1)

        ax.scatter(X[z], y[z], marker='x', alpha=0.9,
                   label=r'observations $y < \tau$', zorder=2)
        ax.scatter(X[~z], y[~z], marker='x', alpha=0.9,
                   label=r'observations $y \geq \tau$', zorder=2)

        ax.axhline(tau, xmin=0.0, xmax=1.0, color='k',
                   linewidth=1.0, linestyle='dashed', zorder=5)

        ax.axvline(x_new, ymin=0.0, ymax=1.0, color=next_loc_color,
                   linewidth=1.0, label="next location", zorder=10)

        ax.set_ylabel(r"$y$")

        ax.legend(loc="upper left")

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("bottom", size=0.6, pad=0.1, sharex=ax)

        ax_top.scatter(X[z], np.ones_like(X[z]), marker='s', alpha=0.9, zorder=2)
        ax_top.scatter(X[~z], np.zeros_like(X[~z]), marker='s', alpha=0.9, zorder=2)

        ax_top.plot(X_grid, tf.sigmoid(model.predict(X_grid)), c='tab:gray',
                    label=r"$\pi_{\theta}(x)$", zorder=1)
        ax_top.axvline(x_new, ymin=0.0, ymax=1.0, color=next_loc_color,
                       linewidth=1.0, zorder=10)

        ax_top.legend(loc="upper left")

        ax_top.set_ylabel(r"$z$")
        ax_top.set_xlabel(r"$x$")

        ax_right = divider.append_axes("right", size=0.6, pad=0.1, sharey=ax)

        sns.ecdfplot(y=y, c='tab:gray', ax=ax_right, zorder=1)

        ax_right.scatter(gamma, tau, c='k', marker='.', zorder=5)
        ax_right.hlines(tau, xmin=0, xmax=gamma, colors='k', linestyles='dashed', linewidth=1.0, zorder=5)
        ax_right.vlines(gamma, ymin=1.1*y_grid.min(), ymax=tau, colors='k', linestyles='dashed', linewidth=1.0, zorder=5)
        # ax_right.annotate(rf"$\gamma={{{gamma:.2f}}}$", xy=(gamma, y.min()),
        #                   xycoords='data', xytext=(5.0, 0.0), textcoords='offset points',
        #                   fontsize="x-small", arrowprops=dict(facecolor='black', arrowstyle='-'))

        ax_right.set_xlabel(r'$\Phi(y)$')
        ax_right.yaxis.set_tick_params(labelleft=False)

        ax_right.set_ylim(1.1*y_grid.min(), 1.1*y_grid.max())

        plt.tight_layout()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"frame_{i:02d}_{context}_{suffix}.{ext}"),
                        dpi=dpi, transparent=transparent)

        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
