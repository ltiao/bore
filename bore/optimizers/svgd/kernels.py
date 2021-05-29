import numpy as np


def _check_length_scale(n_samples, sum_sqr_diff, length_scale=None):
    if length_scale is None:
        h = np.median(sum_sqr_diff)
        length_scale = np.sqrt(.5 * h / np.log(n_samples+1))
    return length_scale


class RadialBasis:

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def value_and_grad(self, X):
        n_samples = X.shape[0]
        diff = np.expand_dims(X, axis=1) - X
        sqr_diff = np.square(diff)
        sum_sqr_diff = np.sum(sqr_diff, axis=-1)
        length_scale = _check_length_scale(n_samples, sum_sqr_diff,
                                           self.length_scale)
        gamma = .5/length_scale**2
        K = np.exp(-gamma*sum_sqr_diff)
        K_grad = 2. * np.sum(gamma * diff * np.expand_dims(K, axis=-1), axis=1)
        return K, K_grad
