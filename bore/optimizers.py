from scipy.optimize import minimize, Bounds
from sklearn.utils import check_random_state


def from_bounds(bounds):

    if isinstance(bounds, Bounds):
        low = bounds.lb
        high = bounds.ub
        dims = len(low)
        assert dims == len(high), "lower and upper bounds sizes do not match"
    else:
        # assumes `bounds` is a list of tuples
        low, high = zip(*bounds)
        dims = len(bounds)

    return (low, high), dims


def multi_start(minimizer_fn=minimize):

    def new_minimizer(fn, bounds, num_starts, num_samples=None,
                      random_state=None, *args, **kwargs):
        """
        Minimize a function from multiple starting points.
        First, the function is evaluated at some number of points that are
        sampled uniformly at random from within the specified bound.
        Then, the minimizer is called on the function using the best points
        from the previous step as the starting points.

        Parameters
        ----------
        num_starts : int
            Number of starting points from which to run the minimizer on the
            function.
        num_samples : int, optional
            Number of points, sampled uniformly at random within the specified
            bound, to evaluate in order to determine the starting points
            (if not specified, defaults to `num_starts`).

        Returns
        -------
        results : list of `OptimizeResult`
            A list of `scipy.optimize.OptimizeResult` objects that encapsulate
            information about the optimization result.
        """
        random_state = check_random_state(random_state)

        assert "x0" not in kwargs, "`x0` should not be specified"
        assert "jac" not in kwargs or kwargs["jac"], "`jac` must be true"

        if num_samples is None:
            num_samples = num_starts

        assert num_samples >= num_starts, \
            "number of random samples (`num_samples`) must be " \
            "greater than number of starting points (`num_starts`)"

        (low, high), dims = from_bounds(bounds)

        # TODO(LT): Allow alternative arbitary generator function callbacks
        # to support e.g. Gaussian sampling, low-discrepancy sequences, etc.
        X_init = random_state.uniform(low=low, high=high, size=(num_samples, dims))

        values, _ = fn(X_init)
        ind = values.argsort()

        results = []
        for i in range(num_starts):
            x_init = X_init[ind[i]]
            result = minimizer_fn(fn, x0=x_init, bounds=bounds, *args, **kwargs)
            results.append(result)

        return results

    return new_minimizer


# DEPRECATED
def _multi_start(minimizer_fn=minimize):

    def new_minimizer(fn, bounds, num_restarts, random_state=None, *args, **kwargs):

        assert "x0" not in kwargs, "`x0` should not be specified"

        if not (num_restarts > 0):
            return []

        results = new_minimizer(fn, bounds, num_restarts-1, random_state,
                                *args, **kwargs)

        # TODO(LT): Allow alternative arbitary generator function callbacks
        # to support e.g. Gaussian sampling, low-discrepancy sequences, etc
        random_state = check_random_state(random_state)
        (low, high), dims = from_bounds(bounds)

        x0 = random_state.uniform(low=low, high=high, size=(dims,))
        result = minimizer_fn(fn, x0=x0, bounds=bounds, *args, **kwargs)
        results.append(result)

        return results

    return new_minimizer


# DEPRECATED
def _random_start(minimizer_fn=minimize):

    def new_minimizer(fn, bounds, num_samples, random_state=None, *args, **kwargs):

        assert "x0" not in kwargs, "`x0` should not be specified"

        # TODO(LT): Allow alternative arbitary generator function callbacks
        # to support e.g. Gaussian sampling, low-discrepancy sequences, etc
        random_state = check_random_state(random_state)
        (low, high), dims = from_bounds(bounds)

        X = random_state.uniform(low=low, high=high, size=(num_samples, dims))
        values, _ = fn(X)
        ind = values.argmin()
        x0 = X[ind]

        return minimizer_fn(fn, x0=x0, bounds=bounds, *args, **kwargs)

    return new_minimizer
