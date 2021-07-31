#!/usr/bin/env python

"""Tests for `bore` package."""

import pytest
import numpy as np

from scipy.optimize import Bounds
from bore.utils.deduplicate import set_diff_2d, pad_unique_random


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 10, 20, 100])
@pytest.mark.parametrize("seed", [0, 42, 8888])
def test_set_diff_2d(dim, seed):

    tol = 1e-8

    n, m = 8, 5

    random_state = np.random.RandomState(seed=seed)

    A = random_state.rand(n, dim)
    B = random_state.rand(m, dim)

    x = random_state.rand(dim)
    y = random_state.rand(dim)

    i = random_state.randint(n-1)
    j = random_state.randint(i+1, n)
    A[i] = x
    A[j] = y

    k = random_state.randint(m-1)
    l = random_state.randint(k+1, m)
    B[k] = x
    B[l] = y

    C = np.vstack((A[:i], A[i+1: j], A[j+1:]))

    np.testing.assert_array_equal(C, set_diff_2d(A, B, tol=tol))


# @pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("dim", [1, 2, 3, 5, 10, 20, 100])
@pytest.mark.parametrize("seed", [0, 42, 8888])
def test_pad_unique_random(dim, seed):

    size = 20
    n, m = 15, 8
    i, j, k = 6, 5, 4

    random_state = np.random.RandomState(seed=seed)
    bounds = Bounds(lb=np.zeros(dim), ub=np.ones(dim))

    C = random_state.rand(k, dim)

    A = np.vstack([np.ones(shape=(i, dim)), random_state.rand(j, dim), C])
    random_state.shuffle(A)

    assert A.shape == (n, dim)

    A_uniq = np.unique(A, axis=0)
    assert len(A) - len(A_uniq) == i - 1

    B = random_state.rand(m, dim)
    B[:k] = C
    random_state.shuffle(B)

    A_minus_B = set_diff_2d(A, B)
    assert len(A) - len(A_minus_B) == k

    mask = np.equal(np.expand_dims(A, axis=1), B).all(axis=-1).any(axis=-1)
    np.testing.assert_array_equal(A[mask].sort(axis=0), C.sort(axis=0))

    D = pad_unique_random(A, size=size, bounds=bounds, B=B,
                          random_state=random_state)
    D_uniq = np.unique(D, axis=0)
    D_minus_B = set_diff_2d(D, B)

    assert D.shape == (size, dim)
    np.testing.assert_array_equal(D, D_uniq)
    np.testing.assert_array_equal(D, D_minus_B)
