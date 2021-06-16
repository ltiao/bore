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

    np.testing.assert_array_equal(bounds.lb, np.zeros(DENSE_DIM))
    np.testing.assert_array_equal(bounds.ub, np.ones(DENSE_DIM))

    assert isinstance(cs_dense.sample_configuration(), CS.Configuration)
    assert isinstance(cs_dense.sample_configuration(size=1), CS.Configuration)

    configs = cs_dense.sample_configuration(size=5)

    assert len(configs) == 5

    for config in configs:
        assert isinstance(config, CS.Configuration)
        assert config.to_array().shape == (DENSE_DIM,)


def test_dense_encoding(config_space, seed):

    ind = 0

    cs_dense = DenseConfigurationSpace(config_space, seed=seed)

    name = cs_dense.get_hyperparameter_by_idx(ind)
    # hp = cs_dense.get_hyperparameter(name)
    assert name == "activation_fn_1"

    dct = {
        'activation_fn_1': 'relu',
        'activation_fn_2': 'tanh',
        'batch_size': 2,
        'dropout_1': 0.39803953082292726,
        'dropout_2': 0.022039062686389176,
        'init_lr': 0,
        'lr_schedule': 'cosine',
        'n_units_1': 5,
        'n_units_2': 1
    }
    config = DenseConfiguration(cs_dense, values=dct)
    array = config.to_array()

    # should always be between 0 and 1
    assert np.less_equal(0., array).all()
    assert np.less_equal(array, 1.).all()

    np.testing.assert_array_almost_equal(array, [0., 1., 1., 0., 0.62500063, 0.44226615, 0.02448785, 0.08333194, 1., 0., 0.91666806, 0.24999917])

    # make sure we recover original dictionary exactly
    config_recon = DenseConfiguration.from_array(cs_dense, array)
    dct_recon = config_recon.get_dictionary()

    assert dct == dct_recon

    # in one-hot encoding scheme, if all entries are "hot" (i.e. nonzero),
    # we should take the argmax
    array[0] = 0.8  # tanh
    array[1] = 0.6  # relu

    config_recon = DenseConfiguration.from_array(cs_dense, array)
    dct_recon = config_recon.get_dictionary()

    assert dct_recon["activation_fn_1"] == "tanh"
