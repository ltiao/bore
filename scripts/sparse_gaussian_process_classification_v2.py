"""Console script for etudes."""
import sys
import click

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from collections import defaultdict
from tqdm import trange

from etudes.utils import (get_distribution_pair, get_kl_weight, to_numpy,
                          get_steps_per_epoch, save_results, DistributionPair)

tfd = tfp.distributions
kernels = tfp.math.psd_kernels

tf.logging.set_verbosity(tf.logging.INFO)

# TODO: add support for option
kernel_cls = kernels.MaternFiveHalves

NUM_TRAIN = 2048
NUM_TEST = 1024
NUM_FEATURES = 1
NUM_INDUCING_POINTS = 32
NUM_QUERY_POINTS = 256

NOISE_VARIANCE = 1e-1
JITTER = 5e-6

QUADRATURE_SIZE = 20
NUM_EPOCHS = 2000
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 256

LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.99

CHECKPOINT_DIR = "models/"
SUMMARY_DIR = "logs/"

CHECKPOINT_PERIOD = 100
SUMMARY_PERIOD = 5
LOG_PERIOD = 1

SEED = 8888


def make_likelihood(f):

    return tfd.Independent(tfd.Bernoulli(logits=f),
                           reinterpreted_batch_ndims=1)


def log_likelihood(y, f):

    p = make_likelihood(f)
    return p.log_prob(y)


@click.command()
@click.argument("name")
@click.option("--num-train", default=NUM_TRAIN, type=int,
              help="Number of training samples")
@click.option("--num-test", default=NUM_TEST, type=int,
              help="Number of test samples")
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
@click.option("-q", "--quadrature-size", default=QUADRATURE_SIZE, type=int,
              help="Quadrature size")
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
def main(name, num_train, num_test, num_features, num_query_points,
         num_inducing_points, noise_variance, num_epochs, batch_size,
         quadrature_size, optimize_variational_posterior, learning_rate,
         beta1, beta2, checkpoint_dir, checkpoint_period, summary_dir,
         summary_period, log_period, jitter, seed):

    random_state = np.random.RandomState(seed)

    # Dataset (training index points)
    p = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
    q = tfd.Normal(loc=0.0, scale=2.0)

    distribution_pair = DistributionPair(p, q)

    X_train, y_train = distribution_pair.make_classification_dataset(num_train, seed=666)
    X_test, y_test = distribution_pair.make_classification_dataset(num_test, seed=666)

    # Model
    # TODO: allow specification of initial values
    amplitude = tfp.util.TransformedVariable(
        1.0, bijector=tfp.bijectors.Exp(), dtype="float64", name='amplitude')
    length_scale = tfp.util.TransformedVariable(
        0.5, bijector=tfp.bijectors.Exp(), dtype="float64", name='length_scale')
    observation_noise_variance = tfp.util.TransformedVariable(
        1e-1, bijector=tfp.bijectors.Exp(), dtype="float64",
        name='observation_noise_variance')

    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)

    initial_inducing_index_points = random_state.choice(X_train.squeeze(),
                                                        num_inducing_points) \
                                                .reshape(-1, num_features)

    inducing_index_points = tf.Variable(initial_inducing_index_points,
                                        name='inducing_index_points')

    variational_loc = tf.Variable(np.zeros(num_inducing_points),
                                  name='variational_loc')
    variational_scale = tf.Variable(np.eye(num_inducing_points),
                                    name='variational_scale')

    x_min, x_max = -5.0, 5.0
    # query index points
    X_pred = np.linspace(x_min, x_max, num_query_points) \
        .reshape(-1, num_features)

    vgp = tfd.VariationalGaussianProcess(
        kernel=kernel,
        index_points=X_pred,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=variational_loc,
        variational_inducing_observations_scale=variational_scale,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter
    )

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                             .shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE) \
                             .batch(batch_size, drop_remainder=True)

    @tf.function
    def elbo(X_batch, y_batch):

        ell = vgp.surrogate_posterior_expected_log_likelihood(
            observation_index_points=X_batch,
            observations=y_batch,
            log_likelihood_fn=log_likelihood,
            quadrature_size=quadrature_size
        )

        kl = vgp.surrogate_posterior_kl_divergence_prior()
        kl_weight = get_kl_weight(num_train, batch_size)

        return ell - kl_weight * kl

    # steps_per_epoch = get_steps_per_epoch(num_train, batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=beta1, beta_2=beta2)

    @tf.function
    def train_step(X_batch, y_batch):

        with tf.GradientTape() as tape:
            nelbo = - elbo(X_batch, y_batch)
            gradients = tape.gradient(nelbo, vgp.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vgp.trainable_variables))

        return nelbo

    timestamp = tf.timestamp()

    keys = ["amplitude", "length_scale", "observation_noise_variance",
            "inducing_index_points", "variational_loc", "variational_scale",
            "timestamp"]
    tensors = [amplitude, length_scale, observation_noise_variance,
               inducing_index_points, variational_loc, variational_scale,
               timestamp]

    history = defaultdict(list)

    with trange(num_epochs, unit="epoch") as range_epochs:

        for epoch in range_epochs:

            for step, (X_batch, y_batch) in enumerate(dataset):

                nelbo = train_step(X_batch, y_batch)

            range_epochs.set_postfix(nelbo=to_numpy(nelbo))

            history["nelbo"].append(to_numpy(nelbo))

            for key, tensor in zip(keys, tensors):

                history[key].append(to_numpy(tensor))

    save_results(history, name, learning_rate, beta1, beta2, num_epochs,
                 summary_dir, seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
