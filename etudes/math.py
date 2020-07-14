import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from noise import pnoise2


def perlin(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=1024,
           repeaty=1024, base=0.0):
    """
    Vectorized light wrapper.

    Examples
    --------

    .. plot::
        :context: close-figs

        # from etudes.math import perlin
        from noise import pnoise2

        step_size = 2.0
        y, x = np.ogrid[0:2:32j, 0:1:32j]
        X, Y = np.broadcast_arrays(x, y)

        Z = np.vectorize(pnoise2)(x, y, octaves=2)
        theta = 2.0 * np.pi * Z

        dx = step_size * np.cos(theta)
        dy = step_size * np.sin(theta)

        fig, ax = plt.subplots(figsize=(10, 8))

        contours = ax.pcolormesh(X, Y, theta)
        ax.quiver(x, y, x + dx, y + dy, alpha=0.8)

        fig.colorbar(contours, ax=ax)

        plt.show()
    """
    return np.vectorize(pnoise2)(x, y, octaves=octaves, persistence=persistence,
                                 lacunarity=lacunarity, repeatx=repeatx,
                                 repeaty=repeaty, base=base)


def expectation_gauss_hermite(fn, normal, quadrature_size):

    def transform(x, loc, scale):

        return np.sqrt(2) * scale * x + loc

    x, weights = np.polynomial.hermite.hermgauss(quadrature_size)
    y = transform(x, normal.loc, normal.scale)

    return tf.reduce_sum(weights * fn(y), axis=-1) / tf.sqrt(np.pi)


def divergence_gauss_hermite(p, q, quadrature_size, under_p=True,
                             discrepancy_fn=tfp.vi.kl_forward):
    """
    Compute D_f[p || q]
        = E_{q(x)}[f(p(x)/q(x))]
        = E_{p(x)}[r(x)^{-1} f(r(x))]          -- r(x) = p(x)/q(x)
        = E_{p(x)}[exp(-log r(x)) g(log r(x))] -- g(.) = f(exp(.))
        = E_{p(x)}[h(x)]                       -- h(x) = exp(-log r(x)) g(log r(x))
    using Gauss-Hermite quadrature assuming p(x) is Gaussian.
    Note `discrepancy_fn` corresponds to function `g`.
    """
    def log_ratio(x):
        return p.log_prob(x) - q.log_prob(x)

    if under_p:

        # TODO: Raise exception if `p` is non-Gaussian.
        w = lambda x: tf.exp(-log_ratio(x))
        normal = p

    else:

        # TODO: Raise exception if `q` is non-Gaussian.
        w = lambda x: 1.0
        normal = q

    def fn(x):
        return w(x) * discrepancy_fn(log_ratio(x))

    return expectation_gauss_hermite(fn, normal, quadrature_size)
