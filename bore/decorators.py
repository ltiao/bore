import tensorflow as tf
import tensorflow_probability as tfp
# import tensorflow.keras.backend as K

from functools import wraps


def negate(fn):

    @wraps(fn)
    def new_fn(*args, **kwargs):
        return -fn(*args, **kwargs)

    return new_fn


def unbatch(fn):

    @wraps(fn)
    def new_fn(input):
        batch_input = tf.expand_dims(input, axis=0)
        batch_output = fn(batch_input)
        output = tf.squeeze(batch_output)
        return output

    return new_fn


def make_value_and_gradient_fn(value_fn):

    @wraps(value_fn)
    def value_and_gradient_fn(input):
        return tfp.math.value_and_gradient(value_fn, input)

    return value_and_gradient_fn


def numpy_outputs(fn):

    @wraps(fn)
    def new_fn(input):

        return tuple(output.numpy() for output in fn(input))

    return new_fn
