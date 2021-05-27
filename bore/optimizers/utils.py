from scipy.optimize import Bounds


def from_bounds(bounds):

    if isinstance(bounds, Bounds):
        low = bounds.lb
        high = bounds.ub
        dims = len(low)
        assert dims == len(high), "lower and upper bounds sizes do not match!"
    else:
        # assumes `bounds` is a list of tuples
        low, high = zip(*bounds)
        dims = len(bounds)

    return (low, high), dims
