import numpy as np

from scipy.optimize import minimize, Bounds
from sklearn.utils import check_random_state


def deduplicate(results, atol=1e-6):

    results_unique = []
    for res in results:
        # - There are more efficient data structures for this, such as a
        #   KD-Tree or Locality Sensitive Hashing (LSH), but these are
        #   premature optimizations at this time
        if any(np.allclose(res_prev.x, res.x, atol=atol)
               for res_prev in results_unique):
            results_unique.append(res)

    return results_unique


def multi_start(minimizer_fn=minimize):

    def new_minimizer(fn, bounds, num_restarts, random_state=None, *args, **kwargs):

        assert "x0" not in kwargs, "`x0` should not be specified"

        if not (num_restarts > 0):
            return []

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

        results = new_minimizer(fn, bounds, num_restarts-1, random_state,
                                *args, **kwargs)
        x0 = random_state.uniform(low=low, high=high, size=(dims,))
        result = minimizer_fn(fn, x0=x0, bounds=bounds, *args, **kwargs)
        results.append(result)

        return results

    return new_minimizer
