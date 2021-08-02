#!/usr/bin/env python

"""Tests for `bore` package."""

import numpy as np
import pytest

from scipy.optimize import Bounds
from bore.models import StackedRecurrentFactory, MaximizableDenseSequential


@pytest.mark.parametrize("seed", [0, 42, 8888])
def test_maximizable_dense_sequential(seed):

    random_state = np.random.RandomState(seed)

    input_dim = 2
    output_dim = 1

    n_layers = 2
    n_units = 32

    n_starts = 5
    n_samples = 1024
    bounds = Bounds(lb=np.zeros(input_dim), ub=np.ones(input_dim))

    model = MaximizableDenseSequential(input_dim=input_dim,
                                       output_dim=output_dim,
                                       num_layers=n_layers,
                                       num_units=n_units)

    X_test = random_state.uniform(low=bounds.lb, high=bounds.ub,
                                  size=(n_samples, input_dim))
    y_test = model.predict(X_test)

    assert y_test.shape == (n_samples, output_dim)

    opt = model.argmax(bounds=bounds,
                       num_starts=n_starts,
                       num_samples=n_samples,
                       method="L-BFGS-B",
                       options=dict(maxiter=1000, ftol=1e-9),
                       print_fn=lambda x: None,
                       random_state=random_state)

    assert opt.x.shape == (input_dim,)

    X_opt = np.expand_dims(opt.x, axis=0)

    assert np.greater_equal(model.predict(X_opt), y_test).all()


@pytest.mark.parametrize("seed", [0, 42, 8888])
def test_stacked_recurrent_factory(seed):

    random_state = np.random.RandomState(seed)

    input_dim = 2
    output_dim = 1

    n_samples = 1024

    n_layers = 2
    n_units = 32
    n_steps = 5

    n_starts = 5
    n_samples = 1024
    bounds = Bounds(lb=np.zeros(input_dim), ub=np.ones(input_dim))

    factory = StackedRecurrentFactory(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=n_layers,
        num_units=n_units,
    )

    X_test = random_state.uniform(low=bounds.lb, high=bounds.ub,
                                  size=(n_samples, input_dim))
    X_test_broad = np.expand_dims(X_test, axis=1)
    X_test_tiled = np.tile(X_test_broad, reps=(n_steps, 1))

    assert X_test_tiled.shape == (n_samples, n_steps, input_dim)

    network1 = factory.build_many_to_many()
    network2 = factory.build_one_to_one(num_steps=n_steps)

    for weights1, weights2 in zip(network1.get_weights(), network2.get_weights()):
        np.testing.assert_array_equal(weights1, weights2)

    Y_test = network1.predict(X_test_tiled)
    y_test = network2.predict(X_test)

    assert Y_test.shape == (n_samples, n_steps, output_dim)
    assert y_test.shape == (n_samples, output_dim)

    np.testing.assert_array_equal(Y_test[::, n_steps - 1, ::], y_test)

    opt = network2.argmax(bounds=bounds,
                          num_starts=n_starts,
                          num_samples=n_samples,
                          method="L-BFGS-B",
                          options=dict(maxiter=1000, ftol=1e-9),
                          print_fn=lambda x: None,
                          random_state=random_state)

    assert opt.x.shape == (input_dim,)

    X_opt = np.expand_dims(opt.x, axis=0)

    assert np.greater_equal(network2.predict(X_opt), y_test).all()
