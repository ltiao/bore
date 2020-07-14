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

NUM_TRAIN = 16
NUM_FEATURES = 1
NOISE_VARIANCE = 1e-1
NUM_EPOCHS = 1000

JITTER = 1e-6

LEARNING_RATE = 1e-2
BETA1 = 0.5
BETA2 = 0.99

CHECKPOINT_DIR = "models/"
CHECKPOINT_PERIOD = 100
SUMMARY_DIR = "logs/"
SUMMARY_PERIOD = 5
LOG_PERIOD = 1

SEED = 8888


def save_results(history, name, seed, learning_rate, beta1, beta2, num_epochs,
                 summary_dir):

    history_df = pd.DataFrame(history).assign(name=name, seed=seed,
                                              learning_rate=learning_rate,
                                              beta1=beta1, beta2=beta2)

    output_dir = Path(summary_dir).joinpath(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_dir.joinpath(f"scalars.{seed:03d}.csv"),
                      index_label="epoch")


@click.command()
@click.argument("name")
@click.option("--num-train", default=NUM_TRAIN, type=int,
              help="Number of training samples")
@click.option("--num-features", default=NUM_FEATURES, type=int,
              help="Number of features (dimensionality)")
@click.option("--noise-variance", default=NOISE_VARIANCE, type=int,
              help="Observation noise variance")
@click.option("-e", "--num-epochs", default=NUM_EPOCHS, type=int,
              help="Number of epochs")
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
def main(name, num_train, num_features, noise_variance, num_epochs,
         learning_rate, beta1, beta2, checkpoint_dir, checkpoint_period,
         summary_dir, summary_period, log_period, jitter, seed):

    # Dataset
    X_train, Y_train = make_dataset(synthetic_sinusoidal, num_train,
                                    num_features, noise_variance)

    # Model hyperparamters
    # TODO: allow specification of initial values
    ln_initial_amplitude = np.float64(0)
    ln_initial_length_scale = np.float64(-1)
    ln_initial_observation_noise_variance = np.float64(-5)

    amplitude = tf.exp(tf.Variable(ln_initial_amplitude), name='amplitude')
    length_scale = tf.exp(tf.Variable(ln_initial_length_scale), name='length_scale')
    observation_noise_variance = tf.exp(tf.Variable(ln_initial_observation_noise_variance,
                                                    name='observation_noise_variance'))

    # Model
    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)
    gp = tfd.GaussianProcess(
        kernel=kernel,
        index_points=X_train,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter
    )
    # Loss (negative log marginal likelihood)
    nll = - gp.log_prob(Y_train)

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1, beta2=beta2)
    train_op = optimizer.minimize(nll, global_step=global_step)

    tf.summary.scalar("loss/nll", nll)

    logger = tf.estimator.LoggingTensorHook(
        tensors=dict(epoch=global_step, nll=nll), every_n_iter=log_period,
        formatter=lambda values: "epoch={epoch:04d}, nll={nll:04f}".format(**values)
    )

    keys = ["nll", "amplitude", "length_scale", "observation_noise_variance"]
    tensors = [nll, amplitude, length_scale, observation_noise_variance]

    fetches = [train_op]
    fetches.extend(tensors)

    history = defaultdict(list)

    with tf.train.MonitoredTrainingSession(
        hooks=[logger],
        checkpoint_dir=os.path.join(checkpoint_dir, name),
        summary_dir=os.path.join(summary_dir, name),
        save_checkpoint_steps=checkpoint_period,
        save_summaries_steps=summary_period
    ) as sess:

        for epoch in range(num_epochs):

            _, *values = sess.run(fetches)

            for key, value in zip(keys, values):

                history[key].append(value)

    save_results(history, name, seed, learning_rate, beta1, beta2, num_epochs,
                 summary_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
