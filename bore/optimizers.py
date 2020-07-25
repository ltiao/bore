import numpy as np

from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.utils import check_random_state


def multi_start(num_restarts, unique=True):

    def decorator(minimizer_fn):

        def new_minimizer(fn, bounds, random_state=None):

            # TODO(LT): Allow alternative arbitary generator function callbacks
            # to support e.g. Gaussian sampling, low-discrepancy sequences, etc
            random_state = check_random_state(random_state)

            if isinstance(bounds, Bounds):
                low = bounds.lb
                high = bounds.ub
                dims = len(low)
                assert dims == len(high), "lower and upper bounds sizes do not match"
            else:
                # assumes `bounds` is a list of tuples
                low, high = zip(*bounds)
                dims = len(bounds)

            x_inits = random_state.uniform(low=low, high=high,
                                           size=(num_restarts, dims))

            results = []
            for x_init in x_inits:
                res = minimizer_fn(fn, x0=x_init, bounds=bounds)

                # TODO(LT): is this a little overbearing? The caller can easily
                #   deal with duplicates themselves.
                #   Further, there are more efficient data structures for this,
                #   such as a KD-Tree or Locality Sensitive Hashing (LSH),
                #   but these are premature optimizations at this time.
                #   Should support other options for `atol`.
                if unique and any(np.allclose(res_prev.x, res.x, atol=1e-6)
                                  for res_prev in results):
                    continue

                results.append(res)

            # TODO(LT): support reduction function callback? e.g. argmin which
            #   is what one ultimately cares about. But perhaps suboptimal
            #   points can be useful as well, e.g. to be queued up for
            #   evaluation by idle workers.
            return results

        return new_minimizer

    return decorator


@multi_start(num_restarts=10, unique=True)
def multi_start_lbfgs_minimizer(fn, x0, bounds):
    """
    Wrapper around SciPy L-BFGS-B minimizer with sensible defaults specified.
    """
    # TODO(LT): L-BFGS-B has its own set of `tol` options so I suspect the
    #   following `tol=1e-8` is completely ignored.
    return minimize(fn, x0=x0, method="L-BFGS-B", jac=True, bounds=bounds,
                    tol=1e-8, options=dict(maxiter=10000))
