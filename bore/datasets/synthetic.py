import numpy as np
import tensorflow_probability as tfp

from sklearn.utils import check_random_state, shuffle as _shuffle

tfd = tfp.distributions


def synthetic_sinusoidal(x):

    return np.sin(12.0*x) + 0.66*np.cos(25.0*x)


def make_regression_dataset(latent_fn=synthetic_sinusoidal):
    """
    Make synthetic dataset.

    Examples
    --------

    Test

    .. plot::
        :context: close-figs

        from bore.datasets.synthetic import synthetic_sinusoidal, make_regression_dataset

        num_train = 64 # nbr training points in synthetic dataset
        num_index_points = 256
        num_features = 1
        observation_noise_variance = 1e-1

        f = synthetic_sinusoidal
        X_pred = np.linspace(-0.6, 0.6, num_index_points).reshape(-1, num_features)

        load_data = make_regression_dataset(f)
        X_train, Y_train = load_data(num_train, num_features,
                                     observation_noise_variance,
                                     x_min=-0.5, x_max=0.5)

        fig, ax = plt.subplots()

        ax.plot(X_pred, f(X_pred), label="true")
        ax.scatter(X_train, Y_train, marker='x', color='k',
                    label="noisy observations")

        ax.legend()

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        plt.show()
    """

    def load_data(num_samples, num_features, noise_variance,
                  x_min=0., x_max=1., squeeze=True, random_state=None):

        rng = check_random_state(random_state)

        eps = noise_variance * rng.randn(num_samples, num_features)

        X = x_min + (x_max - x_min) * rng.rand(num_samples, num_features)
        Y = latent_fn(X) + eps

        if squeeze:
            Y = np.squeeze(Y)

        return X, Y

    return load_data


def make_classification_dataset(X_pos, X_neg, shuffle=False, dtype="float64",
                                random_state=None):

    X = np.vstack([X_pos, X_neg]).astype(dtype)
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    if shuffle:
        X, y = _shuffle(X, y, random_state=random_state)

    return X, y


def make_density_ratio_estimation_dataset(p=None, q=None):

    if p is None:
        p = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
            components_distribution=tfd.Normal(loc=[2.0, -3.0],
                                               scale=[1.0, 0.5]))

    if q is None:
        q = tfd.Normal(loc=0.0, scale=2.0)

    def load_data(num_samples, rate=0.5, dtype="float64", seed=None):

        num_p = int(num_samples * rate)
        num_q = num_samples - num_p

        X_p = p.sample(sample_shape=(num_p, 1), seed=seed).numpy()
        X_q = q.sample(sample_shape=(num_q, 1), seed=seed).numpy()

        X, y = make_classification_dataset(X_p, X_q, dtype=dtype,
                                           random_state=seed)

        return X, y

    return load_data
