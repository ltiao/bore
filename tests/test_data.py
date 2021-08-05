#!/usr/bin/env python

"""Tests for `bore` package."""

import numpy as np
import pytest

from bore.data import MultiFidelityRecord


def test_record():

    input_dim = 2

    gamma = 0.25
    base = 3
    n_rungs = 4

    pad_value = -1.

    seed = 8888
    random_state = np.random.RandomState(seed=seed)

    x1 = random_state.rand(input_dim)
    y11 = random_state.randn()
    y12 = random_state.randn()

    x2 = random_state.rand(input_dim)
    x3 = random_state.rand(input_dim)

    # initialization
    record = MultiFidelityRecord(gamma=gamma)

    assert record.num_rungs() == 0
    assert record.highest_rung() is None

    assert record.rung_sizes() == []
    assert record.size() == 0

    assert record.budgets() == []

    assert record.thresholds() == []

    with pytest.raises(IndexError):

        record.rung_size(0)
        record.rung_size(1)

        record.budget(0)
        record.budget(1)

        record.threshold(0)
        record.threshold(1)

        record.binary_labels(0)
        record.binary_labels(1)

    # adding first element
    record.append(x=x1, y=y11, b=1)

    assert record.num_rungs() == 1
    assert record.highest_rung() == 0

    assert record.rung_sizes() == [1]
    assert record.rung_size(0) == 1
    assert record.size() == 1

    assert record.budgets() == [1]
    assert record.budget(0) == 1

    assert record.thresholds() == [y11]
    assert record.threshold(0) == y11

    assert record.binary_labels(0)

    with pytest.raises(IndexError):

        record.rung_size(1)
        record.budget(1)
        record.threshold(1)
        record.binary_labels(1)

    # adding first element, target for second rung
    record.append(x=x1, y=y12, b=3)

    assert record.num_rungs() == 2
    assert record.highest_rung() == 1

    assert record.rung_sizes() == [1, 1]
    assert record.rung_size(0) == 1
    assert record.rung_size(1) == 1
    assert record.size() == 2

    assert record.budgets() == [1, 3]
    assert record.budget(0) == 1
    assert record.budget(1) == 3

    assert record.thresholds() == [y11, y12]
    assert record.threshold(0) == y11
    assert record.threshold(1) == y12

    assert record.binary_labels(0)
    assert record.binary_labels(1)

    for t in range(n_rungs):
        b = base**t
        record.append(x=x2, y=random_state.randn(), b=b)

        if t > 0:
            record.append(x=x3, y=random_state.randn(), b=b)

    assert record.num_rungs() == 4
    assert record.highest_rung() == 3

    assert record.rung_sizes() == [2, 3, 2, 2]
    assert record.rung_size(0) == 2
    assert record.size() == 9

    assert record.budgets() == [1, 3, 9, 27]
    assert record.budget(0) == 1

    inputs, targets = record.sequences(pad_value=pad_value, binary=False)

    assert record.load_feature_matrix().shape == (3, input_dim)
