"""Console script for etudes."""
import os
import sys
import click

import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import matplotlib as mpl; mpl.use('pgf')
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation
from pathlib import Path
from etudes.datasets import make_dataset, synthetic_sinusoidal

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
golden_size = lambda width: (width, width / GOLDEN_RATIO)

FIG_WIDTH = 10

rc = {
    "figure.figsize": golden_size(FIG_WIDTH),
    "font.serif": ['Times New Roman'],
    "text.usetex": True,
}

sns.set(context="paper", style="ticks", palette="colorblind", font="serif",
        rc=rc)

tf.disable_v2_behavior()

tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# TODO: add support for option
kernel_cls = kernels.ExponentiatedQuadratic

NUM_TRAIN = 512
NUM_FEATURES = 1
NUM_INDUCING_POINTS = 16
NUM_QUERY_POINTS = 256

NOISE_VARIANCE = 1e-1
NUM_EPOCHS = 1000
SUMMARY_DIR = "logs/"

SEED = 42


def wide_data_from_inducing_index_points_history(df):

    df.index.name = "epoch"
    df.columns.name = "inducing index points"

    s = df.stack()
    s.name = 'x'

    return s.reset_index()


def variational_scale_history_from_dataframe(df, num_epochs,
                                             num_inducing_points):

    return df.to_numpy().reshape(num_epochs,
                                 num_inducing_points,
                                 num_inducing_points)


def load_results(name, seed, summary_dir, num_epochs, num_inducing_points, X_q,
                 X_train, Y_train):

    path = Path(summary_dir).joinpath(name)

    y_min, y_max = -3, 3

    inducing_index_points_history_df = pd.read_csv(path.joinpath(f"inducing_index_points.{seed:03d}.csv"), index_col="epoch")
    variational_loc_history_df = pd.read_csv(path.joinpath(f"variational_loc.{seed:03d}.csv"), index_col="epoch")
    variational_scale_history_df = pd.read_csv(path.joinpath(f"variational_scale.{seed:03d}.csv"), index_col="epoch")
    history_df = pd.read_csv(path.joinpath(f"scalars.{seed:03d}.csv"), index_col="epoch")

    amplitude_history = history_df["amplitude"].to_numpy()
    length_scale_history = history_df["length_scale"].to_numpy()
    observation_noise_variance_history = history_df["observation_noise_variance"].to_numpy()

    kernel_history = kernel_cls(amplitude=amplitude_history,
                                length_scale=length_scale_history)

    inducing_index_points_history = np.atleast_3d(inducing_index_points_history_df.to_numpy())
    variational_loc_history = variational_loc_history_df.to_numpy()
    variational_scale_history = variational_scale_history_df.to_numpy() \
                                                            .reshape(num_epochs,
                                                                     num_inducing_points,
                                                                     num_inducing_points)

    segments_min_history = np.dstack(np.broadcast_arrays(inducing_index_points_history, y_min))
    segments_max_history = np.dstack([inducing_index_points_history, variational_loc_history])
    segments_history = np.stack([segments_max_history, segments_min_history], axis=-2)

    vgp_history = tfd.VariationalGaussianProcess(
        kernel=kernel_history,
        index_points=X_q,
        inducing_index_points=inducing_index_points_history,
        variational_inducing_observations_loc=variational_loc_history,
        variational_inducing_observations_scale=variational_scale_history,
        observation_noise_variance=observation_noise_variance_history
    )

    # learning curve

    fig, ax = plt.subplots()

    sns.lineplot(x='epoch', y='nelbo', data=history_df.reset_index(),
                 alpha=0.8, ax=ax)

    # ax.set_xscale("log")

    fig.savefig(os.path.join(summary_dir, f"{name}_learning_curve.png"))
    plt.show()

    # GP hyperparameters

    # g = sns.PairGrid(history_df[["nelbo", "amplitude", "length_scale", "observation_noise_variance"]], hue="nelbo")
    # g = g.map_offdiag(plt.scatter)
    # g.savefig(os.path.join(summary_dir, f"{name}_path.png"))

    # inducing index points

    fig, ax = plt.subplots()

    data = wide_data_from_inducing_index_points_history(inducing_index_points_history_df)

    sns.lineplot(x='x', y="epoch", hue="inducing index points", palette="viridis",
                 sort=False, data=data, alpha=0.8, ax=ax)

    ax.set_xlabel(r'$x$')
    # ax.set_xlim(x_min, x_max)

    fig.savefig(os.path.join(summary_dir, f"{name}.png"))
    plt.show()

    # variational inducing observations scale

    fig, ax = plt.subplots()

    im = ax.imshow(variational_scale_history[-1])

    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$j$")

    plt.show()

    def animate(i):

        im.set_array(variational_scale_history[i])

        return im,

    anim = animation.FuncAnimation(fig, animate, frames=num_epochs,
                                   interval=100, repeat_delay=5, blit=True)
    anim.save(os.path.join(summary_dir, f"{name}.mp4"))
    # , writer='imagemagick'

    # posterior predictive distribution

    vgp_mean = vgp_history.mean()
    vgp_stddev = vgp_history.stddev()

    with tf.Session() as sess:
        vgp_mean_value, vgp_stddev_value = sess.run([vgp_mean, vgp_stddev])

    fig, ax = plt.subplots()

    # ax.plot(X_q, gprm_mean_value[-1])
    # ax.fill_between(np.squeeze(X_q),
    #                 gprm_mean_value[-1] - 2*gprm_stddev_value[-1],
    #                 gprm_mean_value[-1] + 2*gprm_stddev_value[-1], alpha=0.1)

    ax.plot(X_q, vgp_mean_value[-1], label=r"$\mu(x)$")
    ax.fill_between(np.squeeze(X_q),
                    vgp_mean_value[-1] - 2*vgp_stddev_value[-1],
                    vgp_mean_value[-1] + 2*vgp_stddev_value[-1], alpha=0.1,
                    label=r"$2 \sigma(x)$")

    ax.scatter(X_train, Y_train, marker='x', color='k',
               label="noisy observations")

    ax.vlines(inducing_index_points_history[-1],
              ymin=y_min, ymax=variational_loc_history[-1],
              color='k', linewidth=1.0, alpha=0.4, label="inducing inputs")

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_ylim(y_min, y_max)

    ax.legend()

    fig.savefig(os.path.join(summary_dir, f"{name}_posterior_predictive.png"))
    plt.show()

    # posterior predictive distribution 2

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                   gridspec_kw=dict(hspace=0.1))

    ax1.scatter(X_train, Y_train, marker='x', color='k',
                label="noisy observations")

    # ax1.plot(X_q, gprm_mean_value[-1])
    # ax1.fill_between(np.squeeze(X_q),
    #                  gprm_mean_value[-1] - 2*gprm_stddev_value[-1],
    #                  gprm_mean_value[-1] + 2*gprm_stddev_value[-1], alpha=0.1)

    line_mean, = ax1.plot(X_q, vgp_mean_value[-1], color="tab:orange",
                          label=r"$\mu(x)$")
    line_stddev_lower, = ax1.plot(X_q, vgp_mean_value[-1] - 2*vgp_stddev_value[-1],
                                  color="tab:orange", alpha=0.4,
                                  label=r"$2 \sigma(x)$")
    line_stddev_upper, = ax1.plot(X_q, vgp_mean_value[-1] + 2*vgp_stddev_value[-1],
                                  color="tab:orange", alpha=0.4)

    vlines_inducing_index_points = ax1.vlines(inducing_index_points_history[-1],
                                              ymin=y_min, ymax=variational_loc_history[-1],
                                              linewidth=1.0, alpha=0.4,
                                              label="inducing inputs")

    ax1.set_ylabel(r'$y$')
    ax1.set_ylim(y_min, y_max)

    ax1.legend()

    lines_inducing_index_points = ax2.plot(inducing_index_points_history_df.to_numpy(),
                                           range(num_epochs),
                                           color='k', linewidth=1.0, alpha=0.4)

    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel("epoch")

    fig.savefig(os.path.join(summary_dir, f"{name}_posterior_predictive_inducing_index_points.png"))
    fig.savefig(os.path.join(summary_dir, f"{name}_posterior_predictive_inducing_index_points.pgf"))

    def animate(i):

        line_mean.set_data(X_q, vgp_mean_value[i])
        line_stddev_lower.set_data(X_q, vgp_mean_value[i] - 2*vgp_stddev_value[i])
        line_stddev_upper.set_data(X_q, vgp_mean_value[i] + 2*vgp_stddev_value[i])

        vlines_inducing_index_points.set_segments(segments_history[i])

        for j, line in enumerate(lines_inducing_index_points):
            line.set_data(inducing_index_points_history[:i, j], range(i))

        ax2.relim()
        ax2.autoscale_view(scalex=False)

        return line_mean, line_stddev_lower, line_stddev_upper

    anim = animation.FuncAnimation(fig, animate, frames=num_epochs,
                                   interval=100, repeat_delay=5, blit=True)
    anim.save(os.path.join(summary_dir, f"{name}_posterior_predictive.mp4"))


@click.command()
@click.argument("name")
@click.option("--num-train", default=NUM_TRAIN, type=int,
              help="Number of training samples")
@click.option("--num-features", default=NUM_FEATURES, type=int,
              help="Number of features (dimensionality)")
@click.option("--num-query-points", default=NUM_QUERY_POINTS, type=int,
              help="Number of query index points")
@click.option("--num-inducing-points", default=NUM_INDUCING_POINTS, type=int,
              help="Number of inducing index points")
@click.option("--noise-variance", default=NOISE_VARIANCE, type=int,
              help="Observation noise variance")
@click.option("-e", "--num-epochs", default=NUM_EPOCHS, type=int,
              help="Number of epochs")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, num_train, num_features, num_query_points, num_inducing_points,
         noise_variance, num_epochs, summary_dir, seed):

    # Dataset (training index points)
    X_train, Y_train = make_dataset(synthetic_sinusoidal, num_train,
                                    num_features, noise_variance,
                                    x_min=-0.5, x_max=0.5)

    x_min, x_max = -1.0, 1.0
    # query index points
    X_q = np.linspace(x_min, x_max, num_query_points).reshape(-1, num_features)

    load_results(name, seed, summary_dir, num_epochs, num_inducing_points, X_q,
                 X_train, Y_train)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
