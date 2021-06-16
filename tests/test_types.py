#!/usr/bin/env python

"""Tests for `bore` package."""

import numpy as np
import pytest
import ConfigSpace as CS

from bore.plugins.hpbandster.types import DenseConfigurationSpace, DenseConfiguration


@pytest.fixture
def seed():
    return 8888


@pytest.fixture
def config_space(seed):

    cs = CS.ConfigurationSpace(seed=seed)
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("n_units_1", lower=0, upper=5))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("n_units_2", lower=0, upper=5))
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("dropout_1", lower=0, upper=0.9))
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("dropout_2", lower=0, upper=0.9))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
    cs.add_hyperparameter(
        CS.UniformIntegerHyperparameter("init_lr", lower=0, upper=5))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("batch_size", lower=0, upper=3))

    return cs


def test_shapes(config_space, seed):

    SPARSE_DIM = 9
    DENSE_DIM = 12

    cs_dense = DenseConfigurationSpace(config_space, seed=seed)

    assert cs_dense.get_dimensions(sparse=True) == SPARSE_DIM
    assert cs_dense.get_dimensions(sparse=False) == DENSE_DIM

    bounds = cs_dense.get_bounds()

    np.testing.assert_array_almost_equal(bounds.lb, np.zeros(DENSE_DIM))
    np.testing.assert_array_almost_equal(bounds.ub, np.ones(DENSE_DIM))

    assert isinstance(cs_dense.sample_configuration(), CS.Configuration)
    assert isinstance(cs_dense.sample_configuration(size=1), CS.Configuration)

    configs = cs_dense.sample_configuration(size=5)

    assert len(configs) == 5

    for config in configs:
        assert isinstance(config, CS.Configuration)
        assert config.to_array().shape == (DENSE_DIM,)
