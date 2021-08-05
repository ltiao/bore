import numpy as np


def ceil_divide(a, b, *args, **kwargs):
    return - np.floor_divide(-a, b, *args, **kwargs)


def steps_per_epoch(dataset_size, batch_size):
    """
    Compute the number of gradient steps taken in a single epoch (i.e. full
    pass over the data) for a given dataset size and batch size.
    If the batch size does not evenly divide the dataset size, we assume a
    gradient step is still performed on the remaining smaller batch.

    Examples
    --------
    >>> steps_per_epoch(dataset_size=32, batch_size=64)
    1

    >>> steps_per_epoch(dataset_size=64, batch_size=64)
    1

    >>> steps_per_epoch(dataset_size=100, batch_size=64)
    2

    >>> steps_per_epoch(dataset_size=1000, batch_size=64)
    16
    """
    return int(ceil_divide(dataset_size, batch_size))
