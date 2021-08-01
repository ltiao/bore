#!/usr/bin/env python

"""Tests for `bore` package."""

import numpy as np
import pytest

from scipy.optimize import Bounds
from bore.models import StackedRecurrentFactory


@pytest.mark.parametrize("seed", [0, 42, 8888])
def test_stacked_recurrent_factory(seed):

    input_dim = 2
    output_dim = 1

    n_samples = 512

    n_layers = 2
    n_units = 32
    n_steps = 5

    factory = StackedRecurrentFactory(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=n_layers,
        num_units=n_units,
    )

    random_state = np.random.RandomState(seed)

    X_test = random_state.rand(n_samples, input_dim)
    X_test_broad = np.expand_dims(X_test, axis=1)
    X_test_tiled = np.tile(X_test_broad, reps=(n_steps, 1))

    assert X_test_tiled.shape == (n_samples, n_steps, input_dim)

    network1 = factory.build_many_to_many()
    network2 = factory.build_one_to_one(num_steps=n_steps)

    for weights1, weights2 in zip(network1.get_weights(), network2.get_weights()):
        np.testing.assert_array_equal(weights1, weights2)

    Y = network1.predict(X_test_tiled)
    y = network2.predict(X_test)

    assert Y.shape == (n_samples, n_steps, output_dim)
    assert y.shape == (n_samples, output_dim)

    np.testing.assert_array_equal(Y[::, n_steps - 1, ::], y)

    opt = network2.argmax(bounds=Bounds(lb=[0, 0], ub=[1, 1]),
                          num_starts=5,
                          num_samples=1024,
                          method="L-BFGS-B",
                          options=dict(maxiter=1000, ftol=1e-9),
                          random_state=random_state)
    print(opt)
