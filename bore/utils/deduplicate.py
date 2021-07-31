import numpy as np

from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
from ..optimizers.utils import from_bounds


def set_diff_2d(A, B, metric="euclidean", tol=1e-8):
    """
    Find the set difference of two arrays.

    Return the values in A that are not in B.
    """
    not_close = np.greater(cdist(A, B, metric=metric), tol)
    mask = np.all(not_close, axis=-1)
    return A[mask]


def pad_unique_random(A, size, bounds, B=None, metric="euclidean", tol=1e-8,
                      random_state=None):
    """
    Generate samples uniformly at random until we have a batch ``A`` of a
    specified size without duplicates and such that values in ``A`` that are
    not in ``B``.
    """
    random_state = check_random_state(random_state)
    (low, high), dim = from_bounds(bounds)

    A1 = np.unique(A, axis=0)
    if B is not None:
        A1 = set_diff_2d(A1, B, metric=metric, tol=tol)
    size1 = A1.shape[0]
    size2 = size - size1

    if size2 == 0:
        return A1

    A2 = random_state.uniform(low=low, high=high, size=(size2, dim))
    C = np.vstack((A1, A2))
    return pad_unique_random(C, size=size, bounds=bounds, B=B, metric=metric,
                             tol=tol, random_state=random_state)
