#!/usr/bin/env python

"""Tests for `bore` package."""

import pytest
import numpy as np

from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

from bore.optimizers.svgd.base import SVGD
from bore.optimizers.svgd.kernels import RadialBasis


# reference implementation from
# https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/
class ReferenceSVGD:
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=None):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h is None:  # if h is None, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    def update(
        self,
        x0,
        lnprob,
        n_iter=1000,
        stepsize=1e-3,
        bandwidth=None,
        alpha=0.9,
        debug=False,
    ):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError("x0 or lnprob cannot be None!")

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter_ in range(n_iter):
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=bandwidth)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]
            # adagrad
            if iter_ == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (
                    grad_theta ** 2
                )
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta


@pytest.mark.parametrize("n_samples", [1, 4, 16])
@pytest.mark.parametrize("n_features", [1, 2, 64])
@pytest.mark.parametrize("length_scale", [1e-3, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", [42, 8888])
def test_kernel(n_samples, n_features, length_scale, seed):

    random_state = np.random.RandomState(seed)
    X = random_state.rand(n_samples, n_features)

    kernel = RadialBasis(length_scale=length_scale)
    K, K_grad = kernel.value_and_grad(X)

    assert K.shape == (n_samples, n_samples)
    assert K_grad.shape == (n_samples, n_features)

    # compare against scikit-learn implementation as a reference
    gamma = .5/length_scale**2
    np.testing.assert_array_almost_equal(K, rbf_kernel(X, gamma=gamma), decimal=12)

    # compare against implementation from experimental repo associated with
    # original SVGD paper as a reference
    svgd = ReferenceSVGD()
    Kxy, dxkxy = svgd.svgd_kernel(X, h=length_scale)
    np.testing.assert_array_almost_equal(K, Kxy, decimal=10)
    np.testing.assert_array_almost_equal(K_grad, dxkxy, decimal=10)


@pytest.mark.parametrize("n_samples", [1, 4, 16])
@pytest.mark.parametrize("n_features", [1, 2, 64])
@pytest.mark.parametrize("seed", [42, 8888])
def test_kernel_median_trick(n_samples, n_features, seed):

    length_scale = None

    random_state = np.random.RandomState(seed)
    X = random_state.rand(n_samples, n_features)

    kernel = RadialBasis(length_scale=length_scale)
    K, K_grad = kernel.value_and_grad(X)

    assert K.shape == (n_samples, n_samples)
    assert K_grad.shape == (n_samples, n_features)

    # compare against implementation from experimental repo associated with
    # original SVGD paper as a reference
    svgd = ReferenceSVGD()
    Kxy, dxkxy = svgd.svgd_kernel(X, h=length_scale)
    np.testing.assert_array_almost_equal(K, Kxy, decimal=10)
    np.testing.assert_array_almost_equal(K_grad, dxkxy, decimal=10)


@pytest.mark.parametrize("n_iter", [50, 500, 1000])
@pytest.mark.parametrize("batch_size", [4, 16, 64])
@pytest.mark.parametrize("length_scale", [None, 1e-3, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", [42, 8888])
def test_svgd(n_iter, batch_size, length_scale, seed):

    n_features = 2

    step_size = 1e-2
    alpha = .9
    eps = 1e-6
    tau = 1.0

    random_state = np.random.RandomState(seed)
    x_init = random_state.randn(batch_size, n_features)

    mu = np.array([-0.6871, 0.8010])
    precision = np.array([[0.2260, 0.1652],
                          [0.1652, 0.6779]])
    mvn = multivariate_normal(mean=mu, cov=np.linalg.inv(precision))

    def log_prob(x):
        return mvn.logpdf(x)

    def log_prob_grad(x):
        return (mu - x) @ precision

    def func(x):
        return log_prob(x), log_prob_grad(x)

    kernel = RadialBasis(length_scale=length_scale)
    svgd = SVGD(kernel=kernel, n_iter=n_iter, step_size=step_size,
                alpha=alpha, eps=eps, tau=tau)

    x = svgd.optimize(func, batch_size, bounds=[(-3., 3.), (-2., 4.)],
                      random_state=random_state)
    assert x.shape == (batch_size, n_features)

    # Tiny numerical differences in the kernel implementation add up after many
    # iterations. This test defines a dummy kernel that simply plugs in the
    # reference implementation of the kernel to ensure that, all else being
    # equal, our core SVGD algorithm implementation behaves identically to the
    # reference implementation.

    svgd1 = ReferenceSVGD()
    x1 = svgd1.update(x_init, log_prob_grad, n_iter=n_iter, stepsize=step_size,
                      bandwidth=length_scale)

    class DummyKernel:

        def __init__(self, length_scale=1.0):
            self.length_scale = length_scale

        def value_and_grad(self, X):
            return svgd1.svgd_kernel(X, h=self.length_scale)

    kernel = DummyKernel(length_scale=length_scale)
    svgd2 = SVGD(kernel=kernel, n_iter=n_iter, step_size=step_size,
                 alpha=alpha, eps=eps, tau=tau)
    x2 = svgd2.optimize_from_init(func, x_init, bounds=None)
    assert x2.shape == x_init.shape

    np.testing.assert_array_equal(x1, x2)
