import numpy as np

from abc import ABC, abstractmethod
from scipy.optimize import Bounds
from sklearn.utils import check_random_state

from .kernels import RadialBasis
from ..utils import from_bounds


class Distortion(ABC):

    @abstractmethod
    def __call__(self, beta):
        pass


class DistortionConstant(Distortion):

    def __init__(self, c=1.):
        self.c = c

    def __call__(self, beta):
        return self.c


class DistortionExpDecay(Distortion):
    """
    Importance weight or distortion function :math:`omega(\beta)`
    """
    def __init__(self, lambd=1.):
        assert lambd > 0, "lambda must be positive!"
        self.lambd = lambd

    def __call__(self, beta):
        return np.power(beta, -self.lambd)


def rank(a):
    """
    Compute empirical CDF of entries in an array.

    Examples
    --------

    >>> a = np.array([0.4532752, 0.858725 , 0.3792093, 0.6631048, 0.7619765])
    >>> rank(a)
    array([0.4, 1. , 0.2, 0.6, 0.8])

    Handling duplicate values:

    >>> a = np.array([0.4532752, 0.858725 , 0.3792093, 0.3792093, 0.7619765])
    >>> rank(a)
    array([0.6, 1. , 0.4, 0.4, 0.8])

    Comparison to behaviour of ``scipy.stats.percentileofscore``:

    >>> from scipy.stats import percentileofscore
    >>> np.array_equal(100. * rank(a),
    ...                [percentileofscore(a, x, "weak") for x in a])
    True

    """
    assert a.ndim == 1, "only support 1d arrays!"
    return np.less_equal(a, np.expand_dims(a, axis=1)).mean(axis=1)


class SVGD:

    def __init__(self, kernel=RadialBasis(), n_iter=1000, step_size=1e-3,
                 alpha=.9, eps=1e-6, tau=1., distortion=DistortionConstant()):
        self.kernel = kernel
        self.n_iter = n_iter
        self.step_size = step_size
        self.alpha = alpha
        self.eps = eps
        self.tau = tau
        self.distortion = distortion

    def optimize_from_init(self, func, x_init, bounds=None,
                           callback=None):
        """
        Optimize from specified starting points.
        """
        if bounds is not None:
            (low, high), _ = from_bounds(bounds)

        n_init = x_init.shape[0]
        grad_hist = None
        x = x_init.copy()

        for i in range(self.n_iter):

            K, K_grad = self.kernel.value_and_grad(x)

            f, f_grad = func(x)

            zeta = self.distortion(rank(f))
            Zeta = np.expand_dims(zeta, axis=-1)

            grad = (K @ (Zeta * f_grad) + self.tau * K_grad)
            grad /= n_init

            # adadelta / adagrad
            if grad_hist is None:
                grad_hist = grad**2
            else:
                grad_hist *= self.alpha
                grad_hist += (1 - self.alpha) * grad**2

            adj_grad = np.true_divide(grad, self.eps + np.sqrt(grad_hist))
            x += self.step_size * adj_grad

            if bounds is not None:
                x = x.clip(low, high)

            if callback is not None:
                callback(x)

        return x

    def optimize(self, func, batch_size, bounds=None, callback=None,
                 random_state=None):
        """
        Optimize from specified number of uniformly sampled starting points.
        """
        random_state = check_random_state(random_state)
        (low, high), dims = from_bounds(bounds)
        # TODO(LT): Allow alternative arbitary generator function callbacks
        # to support e.g. Gaussian sampling, low-discrepancy sequences, etc.
        x_init = random_state.uniform(low=low, high=high, size=(batch_size, dims))
        return self.optimize_from_init(func, x_init, bounds, callback)
