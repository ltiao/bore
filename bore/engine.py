import numpy as np

from scipy.optimize import minimize
from .optimizers import multi_start


def is_duplicate(x, xs, rtol=1e-5, atol=1e-8):
    # Clever ways of doing this would involve data structs. like KD-trees
    # or locality sensitive hashing (LSH), but these are premature
    # optimizations at this point, especially since the `any` below does lazy
    # evaluation, i.e. is early stopped as soon as anything returns `True`.
    return any(np.allclose(x_prev, x, rtol=rtol, atol=atol) for x_prev in xs)


minimize_multi_start = multi_start(minimizer_fn=minimize)
