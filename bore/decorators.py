import tensorflow as tf

from functools import wraps


def stack(fn):

    @wraps(fn)
    def new_fn(*args):
        return fn(tf.stack(args))

    return new_fn


def unstack(fn):

    @wraps(fn)
    def new_fn(args):
        return fn(*tf.unstack(args, axis=-1))

    return new_fn


def squeeze(axis):

    def squeeze_dec(fn):

        @wraps(fn)
        def new_fn(*args, **kwargs):
            return tf.squeeze(fn(*args, **kwargs), axis=axis)

        return new_fn

    return squeeze_dec


def unbatch(fn):

    @wraps(fn)
    def new_fn(input):
        batch_input = tf.expand_dims(input, axis=0)
        batch_output = fn(batch_input)
        output = tf.squeeze(batch_output, axis=0)
        return output

    return new_fn


def value_and_gradient(value_fn):

    @wraps(value_fn)
    @tf.function
    def value_and_gradient_fn(x):

        # Equivalent to `tfp.math.value_and_gradient(value_fn, x)`, with the
        # only difference that the gradients preserve their `dtype` rather than
        # casting to `tf.float32`, which is problematic for scipy.optimize
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            val = value_fn(x)

        grad = tape.gradient(val, x)

        return val, grad

    return value_and_gradient_fn


def numpy_io(fn):

    @wraps(fn)
    def new_fn(*args):

        new_args = map(tf.convert_to_tensor, args)
        outputs = fn(*new_args)
        new_outputs = [output.numpy() for output in outputs]

        return new_outputs

    return new_fn
