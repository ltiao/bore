import tensorflow as tf

from scipy.stats import truncnorm
from .decorators import unbatch, value_and_gradient, numpy_io, squeeze


def truncated_normal(loc, scale, lower, upper):
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm(a=a, b=b, loc=loc, scale=scale)


def convert(model, transform=tf.identity):
    """
    Builds a callable from a Keras model that takes a single array as input
    (rather than a batch of Tensors), and returns the output value as a scalar
    the and gradient vector as an array.

    This function makes it easy to use methods from ``scipy.optimize`` to
    minimize inputs to a model wrt to its output with option ``jac=True``.

    Parameters
    ----------
    model : a Keras model
        A Keras model, or any batched TensorFlow operation, with output
        dimension 1. More specifically, any operation that takes a Tensor of
        shape ``(None, D)`` as input and outputs Tensor of shape ``(None, 1)``.
    transform : callable, optional
        A function that transforms the output of the model, e.g. negates the
        output for subsequent maximization instead of minimization.

    Returns
    -------
    fn : callable
        A function that takes an array of shape ``(D,)`` as input, and returns
        a pair with shape ``(), (D,)``, consisting of the output scalar and the
        gradient vector.
    """
    @numpy_io  # array input to Tensor and Tensor outputs back to array
    @value_and_gradient  # `(D,) -> ()` to `(D,) -> (), (D,)`
    @squeeze(axis=-1)  # `(D,) -> (1,)` to `(D,) -> ()`
    @unbatch  # `(None, D) -> (None, 1)` to `(D,) -> (1,)`
    def fn(x):
        return transform(model(x))

    return fn
