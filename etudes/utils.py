import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
import h5py

from tensorflow.keras.metrics import binary_accuracy
from .datasets import make_classification_dataset \
    as _make_classification_dataset
from .math import expectation_gauss_hermite, divergence_gauss_hermite \
    as _divergence_gauss_hermite
from pathlib import Path

# shortcuts
tfd = tfp.distributions


def save_hdf5(X_train, y_train, X_test, y_test, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)


def load_hdf5(filename):

    with h5py.File(filename, 'r') as f:
        X_train = np.array(f.get("X_train"))
        y_train = np.array(f.get("y_train"))
        X_test = np.array(f.get("X_test"))
        y_test = np.array(f.get("y_test"))

    return (X_train, y_train), (X_test, y_test)


class DistributionPair:

    def __init__(self, p, q):

        self.p = p
        self.q = q

    @classmethod
    def from_covariate_shift_example(cls):

        train = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=[[-2.0, 3.0], [2.0, 3.0]], scale_diag=[1.0, 2.0])
        )

        test = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=[[0.0, -1.0], [4.0, -1.0]])
        )

        return cls(p=test, q=train)

    def make_dataset(self, num_samples, rate=0.5, dtype="float64", seed=None):

        num_p = int(num_samples * rate)
        num_q = num_samples - num_p

        X_p = self.p.sample(sample_shape=(num_p, 1), seed=seed).numpy()
        X_q = self.q.sample(sample_shape=(num_q, 1), seed=seed).numpy()

        return X_p, X_q

    def make_covariate_shift_dataset(self, class_posterior_fn, num_test,
                                     num_train, threshold=0.5, seed=None):

        num_samples = num_test + num_train
        rate = num_test / num_samples

        X_test, X_train = self.make_dataset(num_samples, rate=rate, seed=seed)
        # TODO: Temporary fix. Need to address issue in `DistributionPair`.
        X_train = X_train.squeeze()
        X_test = X_test.squeeze()
        y_train = (class_posterior_fn(*X_train.T) > threshold).numpy()
        y_test = (class_posterior_fn(*X_test.T) > threshold).numpy()

        return (X_train, y_train), (X_test, y_test)

    def make_classification_dataset(self, num_samples, rate=0.5,
                                    dtype="float64", seed=None):

        X_p, X_q = self.make_dataset(num_samples, rate, dtype, seed)
        X, y = _make_classification_dataset(X_p, X_q, dtype=dtype,
                                            random_state=seed)

        return X, y

    def logit(self, x):

        return self.p.log_prob(x) - self.q.log_prob(x)

    def density_ratio(self, x):

        return tf.exp(self.logit(x))

    def optimal_score(self, x):

        return tf.sigmoid(self.logit(x))

    def optimal_accuracy(self, x_test, y_test):

        # Required when some distributions are inherently `float32` such as
        # the `MixtureSameFamily`.
        # TODO: Add flexibility for whether to cast to `float64`.
        y_pred = tf.cast(tf.squeeze(self.optimal_score(x_test)),
                         dtype=tf.float64)

        return binary_accuracy(y_test, y_pred)

    def kl_divergence(self):

        return tfd.kl_divergence(self.p, self.q)

    def divergence_monte_carlo(self):
        # TODO
        pass

    def make_p_log_prob_estimator(self, logit_estimator):
        """
        Recall log r(x) = log p(x) - log q(x). Then, we have,
            log p(x) = log p(x) - log q(x) + log q(x) = log r(x) + log q(x)
        """
        def p_log_prob_estimator(x):

            return self.q.log_prob(x) + logit_estimator(x)

        return p_log_prob_estimator


class DistributionPairGaussian(DistributionPair):

    def __init__(self, q, p_loc=0.0, p_scale=1.0):

        super(DistributionPairGaussian, self).__init__(
            p=tfd.Normal(loc=p_loc, scale=p_scale), q=q)

    def divergence_gauss_hermite(self, quadrature_size,
                                 discrepancy_fn=tfp.vi.kl_forward):

        return _divergence_gauss_hermite(self.p, self.q,
                                         quadrature_size, under_p=True,
                                         discrepancy_fn=discrepancy_fn)

    def kl_divergence_scaled_distribution(self, logit_estimator, quadrature_size):
        """
        By definition, r(x) q(x) = p(x). That is, we can think of r(x) as the
        scaling factor needed to match q(x) to p(x).
        Let hat{p}(x) = hat{r}(x) q(x). Then, hat{p}(x) ~= p(x) and how good
        this estimate is depends entirely on how well hat{r}(x) estimates r(x).
        This function computes KL[p(x) || hat{p}(x)] >= 0 using Gauss-Hermite
        quadrature assuming p(x) is Gaussian. The lower the better, with
        equality at p(x) == hat{p}(x).
        """
        def fn(x):
            self.logit(x) - logit_estimator(x)

        return expectation_gauss_hermite(fn, self.p, quadrature_size)


qs = {
    "same": tfd.Normal(loc=0.0, scale=1.0),
    "scale_lesser": tfd.Normal(loc=0.0, scale=0.6),
    "scale_greater": tfd.Normal(loc=0.0, scale=2.0),
    "loc_different": tfd.Normal(loc=0.5, scale=1.0),
    "additive": tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.95, 0.05]),
        components_distribution=tfd.Normal(loc=[0.0, 3.0], scale=[1.0, 1.0])),
    "bimodal": tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.4, 0.6]),
        components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
}


def get_distribution_pair(name):

    return DistributionPairGaussian(qs[name])


def get_steps_per_epoch(num_train, batch_size):

    return num_train // batch_size


def get_kl_weight(num_train, batch_size):

    kl_weight = batch_size / num_train

    return kl_weight


def to_numpy(transformed_variable):

    return tf.convert_to_tensor(transformed_variable).numpy()


def inducing_index_points_history_to_dataframe(inducing_index_points_history):
    # TODO: this will fail for `num_features > 1`
    return pd.DataFrame(np.hstack(inducing_index_points_history).T)


def variational_scale_history_to_dataframe(variational_scale_history,
                                           num_epochs):

    a = np.stack(variational_scale_history, axis=0).reshape(num_epochs, -1)
    return pd.DataFrame(a)


def save_results(history, name, learning_rate, beta1, beta2,
                 num_epochs, summary_dir, seed):

    inducing_index_points_history_df = \
        inducing_index_points_history_to_dataframe(history.pop("inducing_index_points"))

    variational_loc_history_df = pd.DataFrame(history.pop("variational_loc"))
    variational_scale_history_df = \
        variational_scale_history_to_dataframe(history.pop("variational_scale"),
                                               num_epochs)

    history_df = pd.DataFrame(history).assign(name=name, seed=seed,
                                              learning_rate=learning_rate,
                                              beta1=beta1, beta2=beta2)

    output_dir = Path(summary_dir).joinpath(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: add flexibility and de-clutter
    inducing_index_points_history_df.to_csv(output_dir.joinpath(f"inducing_index_points.{seed:03d}.csv"), index_label="epoch")
    variational_loc_history_df.to_csv(output_dir.joinpath(f"variational_loc.{seed:03d}.csv"), index_label="epoch")
    variational_scale_history_df.to_csv(output_dir.joinpath(f"variational_scale.{seed:03d}.csv"), index_label="epoch")
    history_df.to_csv(output_dir.joinpath(f"scalars.{seed:03d}.csv"), index_label="epoch")


def preprocess(df):

    new_df = df.assign(timestamp=pd.to_datetime(df.timestamp, unit='s'))
    elapsed_delta = new_df.timestamp - new_df.timestamp.min()

    return new_df.assign(elapsed=elapsed_delta.dt.total_seconds())


def extract_series(df, index="elapsed", column="nelbo"):

    new_df = df.set_index(index)
    series = new_df[column]

    # (0) save last timestamp and value
#     series_final = series.tail(n=1)

    # (1) de-duplicate the values (significantly speed-up
    # subsequent processing)
    # (2) de-duplicate the indices (it is entirely possible
    # for some epoch of two different tasks to complete
    # at the *exact* same time; we take the one with the
    # smaller value)
    # (3) add back last timestamp and value which can get
    # lost in step (1)
    new_series = series.drop_duplicates(keep="first") \
                       .groupby(level=index).min()
#                        .append(series_final)

    return new_series


def merge_stack_runs(series_dict, seed_key="seed", y_key="nelbo",
                     drop_until_all_start=False):

    merged_df = pd.DataFrame(series_dict)

    # fill missing values by propagating previous observation
    merged_df.ffill(axis="index", inplace=True)

    # NaNs can only remain if there are no previous observations
    # i.e. these occur at the beginning rows.
    # drop rows until all runs have recorded observations.
    if drop_until_all_start:
        merged_df.dropna(how="any", axis="index", inplace=True)

    # TODO: Add option to impute with row-wise mean, which looks something like:
    #    (values in Pandas can only be filled column-by-column, so need to
    #     transpose, fillna and transpose back)
    # merged_df = merged_df.T.fillna(merged_df.mean(axis="columns")).T

    merged_df.columns.name = seed_key
    stacked_df = merged_df.stack(level=seed_key)

    stacked_df.name = y_key
    data = stacked_df.reset_index()

    return data


def make_plot_data(names, seeds, summary_dir,
                   process_run_fn=None,
                   extract_series_fn=None,
                   seed_key="seed",
                   y_key="nelbo"):

    base_path = Path(summary_dir)

    if process_run_fn is None:

        def process_run_fn(run_df):
            return run_df

    df_list = []

    for name in names:

        path = base_path.joinpath(name)
        seed_dfs = dict()

        for seed in seeds:

            csv_path = path.joinpath(f"scalars.{seed:03d}.csv")
            seed_df = pd.read_csv(csv_path)

            seed_dfs[seed] = process_run_fn(seed_df)

        if extract_series_fn is not None:

            series_dict = {seed: extract_series_fn(seed_df)
                           for seed, seed_df in seed_dfs.items()}

            name_df = merge_stack_runs(series_dict, seed_key=seed_key,
                                       y_key=y_key).assign(name=name)

        else:

            name_df = pd.concat(seed_dfs.values(), axis="index", sort=True)

        df_list.append(name_df)

    data = pd.concat(df_list, axis="index", sort=True)

    return data
