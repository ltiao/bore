import numpy as np

from .kernels import RadialBasis
from ..utils import from_bounds

from sklearn.utils import check_random_state


class SVGD:

    def __init__(self, kernel=RadialBasis(), n_iter=1000, step_size=1e-3,
                 alpha=.9, eps=1e-6):
        self.kernel = kernel
        self.n_iter = n_iter
        self.step_size = step_size
        self.alpha = alpha
        self.eps = eps

    def optimize_from_init(self, log_prob_grad, x_init, bounds=None):
        """
        Optimize from specified starting points.
        """
        # TODO(LT): Ensure that no particles exceed some user-defined bounds

        n_init = x_init.shape[0]
        grad_hist = None
        x = x_init.copy()

        for i in range(self.n_iter):

            K, K_grad = self.kernel.value_and_grad(x)

            grad = (K @ log_prob_grad(x) + K_grad)
            grad /= n_init

            if grad_hist is None:
                grad_hist = grad**2
            else:
                grad_hist *= self.alpha
                grad_hist += (1 - self.alpha) * grad**2

            adj_grad = np.true_divide(grad, self.eps + np.sqrt(grad_hist))
            x += self.step_size * adj_grad

        return x

    def optimize(self, log_prob_grad, batch_size, bounds=None, random_state=None):
        """
        Optimize from specified number of uniformly sampled starting points.
        """
        random_state = check_random_state(random_state)
        (low, high), dims = from_bounds(bounds)
        # TODO(LT): Allow alternative arbitary generator function callbacks
        # to support e.g. Gaussian sampling, low-discrepancy sequences, etc.
        x_init = random_state.uniform(low=low, high=high, size=(batch_size, dims))
        return self.optimize_from_init(log_prob_grad, x_init, bounds)
