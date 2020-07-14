"""Main module."""
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Identity, Constant

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


def identity_initializer(shape, dtype=None):

    *batch_shape, num_rows, num_columns = shape

    return tf.eye(num_rows, num_columns, batch_shape=batch_shape, dtype=dtype)


class KernelWrapper(Layer):

    # TODO: Support automatic relevance determination
    def __init__(self, input_dim=1, kernel_cls=kernels.ExponentiatedQuadratic,
                 dtype=None, **kwargs):

        super(KernelWrapper, self).__init__(dtype=dtype, **kwargs)

        self.kernel_cls = kernel_cls

        self.log_amplitude = self.add_weight(
            name="log_amplitude",
            initializer="zeros", dtype=dtype)

        self.log_length_scale = self.add_weight(
            name="log_length_scale",
            initializer="zeros", dtype=dtype)

        self.log_scale_diag = self.add_weight(
            name="log_scale_diag", shape=(input_dim,),
            initializer="zeros", dtype=dtype)

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):

        base_kernel = self.kernel_cls(
            amplitude=tf.exp(self.log_amplitude),
            length_scale=tf.exp(self.log_length_scale))

        return kernels.FeatureScaled(base_kernel,
                                     scale_diag=tf.exp(self.log_scale_diag))


class VariationalGaussianProcessScalar(tfp.layers.DistributionLambda):

    def __init__(self, kernel_wrapper, num_inducing_points,
                 inducing_index_points_initializer, mean_fn=None, jitter=1e-6,
                 convert_to_tensor_fn=tfd.Distribution.sample, **kwargs):

        def make_distribution(x):

            return VariationalGaussianProcessScalar.new(
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

        super(VariationalGaussianProcessScalar, self).__init__(
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


class GaussianProcessLayer(Layer):

    def __init__(self, units, kernel_provider, num_inducing_points=64,
                 mean_fn=None, jitter=1e-6, **kwargs):

        self.units = units  # TODO: Maybe generalize to `event_shape`?
        self.num_inducing_points = num_inducing_points
        self.kernel_provider = kernel_provider
        self.mean_fn = mean_fn
        self.jitter = jitter

        super(GaussianProcessLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        input_dim = input_shape[-1]

        # TODO: Fix initialization!
        self.inducing_index_points = self.add_weight(
            name="inducing_index_points",
            shape=(self.units, self.num_inducing_points, input_dim),
            initializer=tf.keras.initializers.RandomUniform(-1, 1),
            trainable=True)

        self.variational_loc = self.add_weight(
            name="variational_inducing_observations_loc",
            shape=(self.units, self.num_inducing_points),
            initializer='zeros', trainable=True)

        self.variational_scale = self.add_weight(
            name="variational_inducing_observations_scale",
            shape=(self.units,
                   self.num_inducing_points,
                   self.num_inducing_points),
            initializer=identity_initializer, trainable=True)

        super(GaussianProcessLayer, self).build(input_shape)

    def call(self, x):

        base = tfd.VariationalGaussianProcess(
            kernel=self.kernel_provider.kernel,
            index_points=x,
            inducing_index_points=self.inducing_index_points,
            variational_inducing_observations_loc=self.variational_loc,
            variational_inducing_observations_scale=self.variational_scale,
            mean_fn=self.mean_fn,
            predictive_noise_variance=1e-1,  # TODO: what does this mean in the non-Gaussian likelihood context? Should keep it zero.
            jitter=self.jitter
        )

        # sum KL divergence between `units` independent processes
        self.add_loss(tf.reduce_sum(base.surrogate_posterior_kl_divergence_prior()))

        bijector = tfp.bijectors.Transpose(rightmost_transposed_ndims=2)
        qf = tfd.TransformedDistribution(
            tfd.Independent(base, reinterpreted_batch_ndims=1),
            bijector=bijector)

        return qf.sample()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def gp_sample_custom(gp, n_samples, seed=None):

    gp_marginal = gp.get_marginal_distribution()

    batch_shape = tf.ones(gp_marginal.batch_shape.rank, dtype=tf.int32)
    event_shape = gp_marginal.event_shape

    sample_shape = tf.concat([[n_samples], batch_shape, event_shape], axis=0)

    base_samples = gp_marginal.distribution.sample(sample_shape, seed=seed)
    gp_samples = gp_marginal.bijector.forward(base_samples)

    return gp_samples


def dataframe_from_gp_samples(gp_samples_arr, X_q, amplitude, length_scale,
                              n_samples):

    names = ["sample", "amplitude", "length_scale", "index_point"]

    v = [list(map(r"$i={}$".format, range(n_samples))),
         amplitude.squeeze(), length_scale.squeeze(), X_q.squeeze()]

    index = pd.MultiIndex.from_product(v, names=names)

    d = pd.DataFrame(gp_samples_arr.ravel(), index=index, columns=["function_value"])

    return d.reset_index()


def dataframe_from_gp_summary(gp_mean_arr, gp_stddev_arr, amplitude,
                              length_scale, index_point):

    names = ["amplitude", "length_scale", "index_point"]
    v = [amplitude.squeeze(), length_scale.squeeze(), index_point.squeeze()]

    index = pd.MultiIndex.from_product(v, names=names)

    d1 = pd.DataFrame(gp_mean_arr.ravel(), index=index, columns=["mean"])
    d2 = pd.DataFrame(gp_stddev_arr.ravel(), index=index, columns=["stddev"])

    data = pd.merge(d1, d2, left_index=True, right_index=True)

    return data.reset_index()
