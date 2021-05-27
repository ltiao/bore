#!/usr/bin/env python

"""Tests for `bore` package."""

import pytest
import numpy as np

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
from bore.optimizers.svgd.kernels import RadialBasis


# reference implementation from
# https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/
class SVGD:
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
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
        bandwidth=-1,
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
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
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
    gamma = 0.5 / length_scale ** 2
    assert np.allclose(K, rbf_kernel(X, gamma=gamma))

    # compare against implementation from experimental repo associated with
    # original SVGD paper as a reference
    svgd = SVGD()
    Kxy, dxkxy = svgd.svgd_kernel(X, h=length_scale)
    assert np.allclose(K, Kxy)
    assert np.allclose(K_grad, dxkxy)
