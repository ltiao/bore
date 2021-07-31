import numpy as np


def ceil_divide(a, b, *args, **kwargs):
    return - np.floor_divide(-a, b, *args, **kwargs)
