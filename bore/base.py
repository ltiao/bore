import tensorflow as tf

from scipy.stats import truncnorm
from .decorators import unbatch, value_and_gradient, numpy_io, squeeze


def convert(model, transform=tf.identity):
    """
    Given a Keras model, builds a callable that takes a single array as input
    (rather than a batch of Tensors) and returns a pair containing the output
    value (a scalar) and the gradient vector (an array).

    This function makes it easy to use optimization methods
    from ``scipy.optimize`` to minimize inputs to a model wrt to its output
    using the option ``jac=True``.

    Parameters
    ----------
    model : a Keras model
        A Keras model, or any batched TensorFlow operation, with output
        dimension 1. More precisely, any operation that takes a Tensor of
        shape ``(None, D)`` as input and returns as output a Tensor of
        shape ``(None, 1)``.
    transform : callable, optional
        A function that transforms the output of the model, e.g. negating the
        output effectively maximizes instead of minimizes it.

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


def truncated_normal(loc, scale, lower, upper):
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm(a=a, b=b, loc=loc, scale=scale)


def maybe_distort(loc, distortion=None, bounds=None, random_state=None,
                  print_fn=print):

    if distortion is None:
        return loc

    assert bounds is not None, "must specify bounds!"
    ret = truncated_normal(loc=loc,
                           scale=distortion,
                           lower=bounds.lb,
                           upper=bounds.ub).rvs(random_state=random_state)
    print_fn(f"Suggesting x={ret} (after applying distortion={distortion:.3E})")

    return ret
