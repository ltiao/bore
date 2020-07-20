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


def value_and_gradient(value_fn):

    @wraps(value_fn)
    def value_and_gradient_fn(x):

        # Equivalent to `tfp.math.value_and_gradient(value_fn, x)`, with the
        # only difference that the gradients preserve their `dtype`
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            val = value_fn(x)

        grad = tape.gradient(val, x)

        return val, grad

    return value_and_gradient_fn


def numpy_io(fn):

    @wraps(fn)
    def new_fn(input):

        input_tensor = tf.convert_to_tensor(input)
        output_tensor = fn(input_tensor)

        return [output.numpy() for output in output_tensor]

    return new_fn
