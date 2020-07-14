"""Console script for etudes."""
import os
import sys
import click

import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from collections import defaultdict
from pathlib import Path

from etudes.datasets import make_dataset, synthetic_sinusoidal

tf.disable_v2_behavior()

tfd = tfp.distributions
kernels = tfp.math.psd_kernels

tf.logging.set_verbosity(tf.logging.INFO)

# TODO: add support for option
kernel_cls = kernels.ExponentiatedQuadratic

NUM_TRAIN = 512
NUM_FEATURES = 1
NUM_INDUCING_POINTS = 16
NUM_QUERY_POINTS = 256

JITTER = 1e-6

NOISE_VARIANCE = 1e-1
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.99

CHECKPOINT_DIR = "models/"
SUMMARY_DIR = "logs/"

CHECKPOINT_PERIOD = 100
SUMMARY_PERIOD = 5
LOG_PERIOD = 1

SEED = 8888

SHUFFLE_BUFFER_SIZE = 256


def inducing_index_points_history_to_dataframe(inducing_index_points_history):
    # TODO: this will fail for `num_features > 1`
    return pd.DataFrame(np.hstack(inducing_index_points_history).T)


def variational_scale_history_to_dataframe(variational_scale_history,
                                           num_epochs):

    a = np.stack(variational_scale_history, axis=0).reshape(num_epochs, -1)
    return pd.DataFrame(a)


def save_results(history, name, learning_rate, beta1, beta2,
                 num_epochs, summary_dir, seed):

    inducing_index_points_history_df = \
        inducing_index_points_history_to_dataframe(history.pop("inducing_index_points"))

    variational_loc_history_df = pd.DataFrame(history.pop("variational_loc"))
    variational_scale_history_df = \
        variational_scale_history_to_dataframe(history.pop("variational_scale"),
                                               num_epochs)

    history_df = pd.DataFrame(history).assign(name=name, seed=seed,
                                              learning_rate=learning_rate,
                                              beta1=beta1, beta2=beta2)

    output_dir = Path(summary_dir).joinpath(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: add flexibility and de-clutter
    inducing_index_points_history_df.to_csv(output_dir.joinpath(f"inducing_index_points.{seed:03d}.csv"), index_label="epoch")
    variational_loc_history_df.to_csv(output_dir.joinpath(f"variational_loc.{seed:03d}.csv"), index_label="epoch")
    variational_scale_history_df.to_csv(output_dir.joinpath(f"variational_scale.{seed:03d}.csv"), index_label="epoch")
    history_df.to_csv(output_dir.joinpath(f"scalars.{seed:03d}.csv"), index_label="epoch")


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
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int,
              help="Batch size")
@click.option("--optimize-variational-posterior", is_flag=True,
              help="Optimize variational posterior else compute analytically.")
@click.option("--learning-rate", default=LEARNING_RATE,
              type=float, help="Learning rate")
@click.option("--beta1", default=BETA1,
              type=float, help="Beta 1 optimizer parameter")
@click.option("--beta2", default=BETA2,
              type=float, help="Beta 2 optimizer parameter")
@click.option("--checkpoint-dir", default=CHECKPOINT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Model checkpoint directory")
@click.option("--checkpoint-period", default=CHECKPOINT_PERIOD, type=int,
              help="Interval (number of epochs) between checkpoints")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory")
@click.option("--summary-period", default=SUMMARY_PERIOD, type=int,
              help="Interval (number of epochs) between summary saves")
@click.option("--log-period", default=LOG_PERIOD, type=int,
              help="Interval (number of epochs) between logging metrics")
@click.option("--jitter", default=JITTER, type=float, help="Jitter")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, num_train, num_features, num_query_points, num_inducing_points,
         noise_variance, num_epochs, batch_size, optimize_variational_posterior,
         learning_rate, beta1, beta2, checkpoint_dir, checkpoint_period,
         summary_dir, summary_period, log_period, jitter, seed):

    random_state = np.random.RandomState(seed)

    # Dataset (training index points)
    X_train, Y_train = make_dataset(synthetic_sinusoidal, num_train,
                                    num_features, noise_variance,
                                    x_min=-0.5, x_max=0.5)

    x_min, x_max = -1.0, 1.0
    # query index points
    X_q = np.linspace(x_min, x_max, num_query_points).reshape(-1, num_features)

    # Model
    # TODO: allow specification of initial values
    ln_initial_amplitude = np.float64(0)
    ln_initial_length_scale = np.float64(-1)
    ln_initial_observation_noise_variance = np.float64(-5)

    amplitude = tf.exp(tf.Variable(ln_initial_amplitude), name='amplitude')
    length_scale = tf.exp(tf.Variable(ln_initial_length_scale), name='length_scale')
    observation_noise_variance = tf.exp(tf.Variable(ln_initial_observation_noise_variance,
                                                    name='observation_noise_variance'))

    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)

    # Option 1
    initial_inducing_index_points = random_state.choice(X_train.squeeze(),
                                                        num_inducing_points) \
                                                .reshape(-1, num_features)

    # Option 2
    # initial_inducing_index_points = random_state.uniform(x_min, x_max,
    #                                                      num_inducing_points) \
    #                                             .reshape(-1, num_features)

    inducing_index_points = tf.Variable(initial_inducing_index_points,
                                        name='inducing_index_points')

    # Option 3
    # initial_inducing_index_points = np.zeros(shape=(num_inducing_points, num_features))
    # initial_inducing_index_points = np.cumsum(np.insert(0.15 * np.ones(num_inducing_points-1), 0, -1.2)).reshape(-1, num_features)
    # inducing_index_points = tf.cumsum(tf.Variable(initial_inducing_index_points,
    #                                               name='inducing_index_points'), axis=0)

    if optimize_variational_posterior:
        variational_loc = tf.Variable(np.zeros(num_inducing_points),
                                      name='variational_loc')
        variational_scale = tf.Variable(np.eye(num_inducing_points),
                                        name='variational_scale')
    else:
        # Compute optimal variational parameters
        variational_loc, variational_scale = \
            tfd.VariationalGaussianProcess.optimal_variational_posterior(
                kernel=kernel,
                inducing_index_points=inducing_index_points,
                observation_index_points=X_train,
                observations=Y_train,
                observation_noise_variance=observation_noise_variance,
                jitter=jitter
            )

    vgp = tfd.VariationalGaussianProcess(
        kernel=kernel,
        index_points=X_q,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=variational_loc,
        variational_inducing_observations_scale=variational_scale,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter
    )

    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
                             .shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE) \
                             .batch(batch_size, drop_remainder=True)
    iterator = tf.data.make_initializable_iterator(dataset)
    X_batch, Y_batch = iterator.get_next()

    nelbo = vgp.variational_loss(observation_index_points=X_batch,
                                 observations=Y_batch,
                                 kl_weight=batch_size/num_train)

    steps_per_epoch = num_train // batch_size

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1, beta2=beta2)
    train_op = optimizer.minimize(nelbo)

    timestamp = tf.timestamp()

    keys = ["nelbo", "amplitude", "length_scale", "observation_noise_variance",
            "inducing_index_points", "variational_loc", "variational_scale",
            "timestamp"]
    tensors = [nelbo, amplitude, length_scale, observation_noise_variance,
               inducing_index_points, variational_loc, variational_scale,
               timestamp]

    fetches = [train_op]
    fetches.extend(tensors)

    history = defaultdict(list)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            # (re)initialize dataset iterator
            sess.run(iterator.initializer)

            for step in range(steps_per_epoch):

                _, *values = sess.run(fetches)

            for key, value in zip(keys, values):

                history[key].append(value)

    save_results(history, name, learning_rate, beta1, beta2, num_epochs,
                 summary_dir, seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
