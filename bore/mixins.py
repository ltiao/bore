import tensorflow as tf

from .optimizers import minimize_multi_start
from .optimizers.svgd import SVGD
from .optimizers.svgd.kernels import RadialBasis
from .base import convert


class MaximizableMixin:

    def __init__(self, transform=tf.identity, *args, **kwargs):
        super(MaximizableMixin, self).__init__(*args, **kwargs)
        # negate to turn into minimization problem for ``scipy.optimize``
        # interface
        self._func_min = convert(self, transform=lambda u: transform(-u))

    def maxima(self, bounds, num_starts=5, num_samples=1024, method="L-BFGS-B",
               options=dict(maxiter=1000, ftol=1e-9), random_state=None):
        return minimize_multi_start(self._func_min, bounds=bounds,
                                    num_starts=num_starts,
                                    num_samples=num_samples,
                                    random_state=random_state,
                                    method=method, jac=True, options=options)

    def argmax(self, bounds, print_fn=print, filter_fn=lambda res: True,
               *args, **kwargs):

        # Equivalent to:
        # res_best = min(filter(lambda res: res.success or res.status == 1,
        #                       self.maxima(bounds, *args, **kwargs)),
        #                key=lambda res: res.fun)
        res_best = None
        for i, res in enumerate(self.maxima(bounds, *args, **kwargs)):

            print_fn(f"[Maximum {i+1:02d}: value={res.fun:.3f}] "
                     f"success: {res.success}, "
                     f"iterations: {res.nit:02d}, "
                     f"status: {res.status} ({res.message})")

            # TODO(LT): Create Enum type for these status codes `status == 1`
            # signifies maximum iteration reached, which we don't want to
            # treat as a failure condition.
            if (res.success or res.status == 1) and filter_fn(res):
                if res_best is None or res.fun < res_best.fun:
                    res_best = res

        return res_best


class BatchMaximizableMixin(MaximizableMixin):

    def __init__(self, transform=tf.identity, *args, **kwargs):
        super(BatchMaximizableMixin, self).__init__(transform=transform,
                                                    *args, **kwargs)
        # maximization problem for SVGD
        self._func_max = convert(self, transform=transform)

    def argmax_batch(self, batch_size, bounds, length_scale=None, n_iter=1000,
                     step_size=1e-3, alpha=.9, eps=1e-6, random_state=None):

        # def log_prob_grad(x):
        #     _, grad = self._func_max(x)
        #     return grad

        kernel = RadialBasis(length_scale=length_scale)
        svgd = SVGD(kernel=kernel, n_iter=n_iter, step_size=step_size,
                    alpha=alpha, eps=eps)

        return svgd.optimize(self._func_max, batch_size, bounds=bounds,
                             random_state=random_state)
