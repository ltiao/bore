import numpy as np


def svgd(log_prob_grad, kernel, x_init, n_iter=1000, step_size=1e-3, alpha=.9,
         eps=1e-6):

    # TODO(LT): What ensuring that no particles exceed some user-defined bounds?
    n_init = x_init.shape[0]
    grad_hist = None
    x = x_init.copy()

    for i in range(n_iter):

        K, K_grad = kernel.value_and_grad(x)

        grad = K @ log_prob_grad(x) + K_grad
        grad /= n_init

        if grad_hist is None:
            grad_hist = grad**2
        else:
            grad_hist *= alpha
            grad_hist += (1 - alpha) * grad**2

        adj_grad = np.true_divide(grad, eps + np.sqrt(grad_hist))
        x += step_size * adj_grad

    return x
