import numpy as np
import tensorflow as tf

from scipy.optimize import minimize, OptimizeResult
from sklearn.utils import check_random_state

from .base import convert
from .optimizers.utils import from_bounds
from .optimizers.svgd import SVGD
from .optimizers.svgd.base import DistortionConstant, DistortionExpDecay
from .optimizers.svgd.kernels import RadialBasis


class MaximizableMixin:

    def __init__(self, transform=tf.identity, *args, **kwargs):
        super(MaximizableMixin, self).__init__(*args, **kwargs)
        # negate to turn into minimization problem for ``scipy.optimize``
        # interface
        self._func_min = convert(self, transform=lambda u: transform(-u))

    def maxima(self, bounds, num_starts=5, num_samples=1024, method="L-BFGS-B",
               options=dict(maxiter=1000, ftol=1e-9), print_fn=print,
               random_state=None):

        # TODO(LT): Deprecated until minor bug fixed.
        # return minimize_multi_start(self._func_min, bounds=bounds,
        #                             num_starts=num_starts,
        #                             num_samples=num_samples,
        #                             random_state=random_state,
        #                             method=method, jac=True, options=options)

        random_state = check_random_state(random_state)

        assert num_samples is not None, "`num_samples` must be specified!"
        assert num_samples > 0, "`num_samples` must be positive integer!"

        assert num_starts is not None, "`num_starts` must be specified!"
        assert num_starts >= 0, "`num_starts` must be nonnegative integer!"

        assert num_samples >= num_starts, \
            "number of random samples (`num_samples`) must be " \
            "greater than number of starting points (`num_starts`)"

        (low, high), dim = from_bounds(bounds)

        # TODO(LT): Allow alternative arbitary generator function callbacks
        # to support e.g. Gaussian sampling, low-discrepancy sequences, etc.
        X_init = random_state.uniform(low=low, high=high, size=(num_samples, dim))
        z_init = self.predict(X_init).squeeze(axis=-1)
        # the function to minimize is negative of the classifier output
        f_init = - z_init

        results = []
        if num_starts > 0:
            ind = np.argpartition(f_init, kth=num_starts-1, axis=None)
            for i in range(num_starts):
                x0 = X_init[ind[i]]
                result = minimize(self._func_min, x0=x0, method=method,
                                  jac=True, bounds=bounds, options=options)
                results.append(result)
                # TODO(LT): Make this message a customizable option.
                print_fn(f"[Maximum {i+1:02d}: value={result.fun:.3f}] "
                         f"success: {result.success}, "
                         f"iterations: {result.nit:02d}, "
                         f"status: {result.status} ({result.message})")
        else:
            i = np.argmin(f_init, axis=None)
            result = OptimizeResult(x=X_init[i], fun=f_init[i], success=True)
            results.append(result)

        return results

    def argmax(self, bounds, filter_fn=lambda res: True, *args, **kwargs):

        # Equivalent to:
        # res_best = min(filter(lambda res: res.success or res.status == 1,
        #                       self.maxima(bounds, *args, **kwargs)),
        #                key=lambda res: res.fun)
        res_best = None
        for i, res in enumerate(self.maxima(bounds, *args, **kwargs)):
            # TODO(LT): Create Enum type for these status codes.
            # `status == 1` signifies maximum iteration reached, which we don't
            # want to treat as a failure condition.
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
                     step_size=1e-3, alpha=.9, eps=1e-6, tau=1.0, lambd=None,
                     random_state=None):

        distortion = DistortionConstant() if lambd is None \
            else DistortionExpDecay(lambd=lambd)

        # def log_prob_grad(x):
        #     _, grad = self._func_max(x)
        #     return grad

        kernel = RadialBasis(length_scale=length_scale)
        svgd = SVGD(kernel=kernel, n_iter=n_iter, step_size=step_size,
                    alpha=alpha, eps=eps, tau=tau, distortion=distortion)

        return svgd.optimize(self._func_max, batch_size, bounds=bounds,
                             random_state=random_state)
