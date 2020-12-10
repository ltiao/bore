import tensorflow as tf

from scipy.optimize import minimize

from .optimizers import multi_start
from .engine import convert


minimize_multi_start = multi_start(minimizer_fn=minimize)


class MinimizableMixin:

    def __init__(self, transform=tf.identity, *args, **kwargs):
        super(MinimizableMixin, self).__init__(*args, **kwargs)
        self.func = convert(self, transform=transform)

    def minima(self, bounds, num_start_points=3, method="L-BFGS-B",
               options=dict(maxiter=200, ftol=1e-9), random_state=None):

        return minimize_multi_start(self.func, bounds=bounds,
                                    num_restarts=num_start_points,
                                    random_state=random_state,
                                    method=method, jac=True, options=options)

    def argmin(self, bounds, print_fn=print, *args, **kwargs):

        # Equivalent to:
        # res_best = min(filter(lambda res: res.success or res.status == 1,
        #                       self.minima(bounds, *args, **kwargs)),
        #                key=lambda res: res.fun)
        res_best = None
        for j, res in enumerate(self.minima(bounds, *args, **kwargs)):

            print_fn(f"[Maximum {j+1:02d}: value={res.fun:.3f}] "
                     f"success: {res.success}, "
                     f"iterations: {res.nit:02d}, "
                     f"status: {res.status} ({res.message})")

            # TODO(LT): Create Enum type for these status codes
            # status == 1 signifies maximum iteration reached, which we don't
            # want to treat as a failure condition.
            if (res.success or res.status == 1):
                # and not self.record.is_duplicate(res.x):
                if res_best is None or res.fun < res_best.fun:
                    res_best = res

        return res_best.x
