import numpy as np
import tensorflow as tf

from scipy.optimize import minimize

from tensorflow.keras.losses import BinaryCrossentropy

from .types import DenseConfigurationSpace, DenseConfiguration
from .models import DenseSequential
from .decorators import unbatch, value_and_gradient, numpy_io
from .optimizers import multi_start

from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.base_config_generator import base_config_generator


minimize_multi_start = multi_start(minimizer_fn=minimize)


def is_duplicate(x, xs, rtol=1e-5, atol=1e-8):
    # Clever ways of doing this would involve data structs. like KD-trees
    # or locality sensitive hashing (LSH), but these are premature
    # optimizations at this point, especially since the `any` below does lazy
    # evaluation, i.e. is early stopped as soon as anything returns `True`.
    return any(np.allclose(x_prev, x, rtol=rtol, atol=atol) for x_prev in xs)


class BORE(HyperBand):

    def __init__(self, config_space, eta=3, min_budget=0.01, max_budget=1,
                 gamma=None, num_random_init=10, random_rate=0.25,
                 num_restarts=10, batch_size=64, num_steps_per_iter=1000,
                 optimizer="adam", num_layers=2, num_units=32,
                 activation="relu", normalize=True, method="L-BFGS-B",
                 max_iter=100, ftol=1e-2, seed=None, **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = DRE(config_space=config_space,
                 gamma=gamma, num_random_init=num_random_init,
                 random_rate=random_rate, num_restarts=num_restarts,
                 batch_size=batch_size, num_steps_per_iter=num_steps_per_iter,
                 optimizer=optimizer, num_layers=num_layers,
                 num_units=num_units, activation=activation,
                 normalize=normalize, method=method, max_iter=max_iter,
                 ftol=ftol, seed=seed)
        # (LT): Note this is using the *grandparent* class initializer to
        # replace the config_generator!
        super(HyperBand, self).__init__(config_generator=cg, **kwargs)

        # (LT): the following had to be copy-pasted as it is hard to
        # follow the DRY design philosophy within the HpBandSter framework...

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))

        conf = {
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'gamma': gamma,
            'num_random_init': num_random_init,
            'seed': seed
        }
        self.config.update(conf)


class DRE(base_config_generator):
    """
    class to implement random sampling from a ConfigSpace
    """
    def __init__(self, config_space, gamma=1/3, num_random_init=10,
                 random_rate=0.25, num_restarts=3, batch_size=64,
                 num_steps_per_iter=1000, optimizer="adam", num_layers=2,
                 num_units=32, activation="relu", normalize=True,
                 method="L-BFGS-B", max_iter=100, ftol=1e-2, seed=None,
                 **kwargs):

        super(DRE, self).__init__(**kwargs)

        assert 0. <= gamma < 1., "`gamma` must be in [0, 1)"
        assert 0. <= random_rate < 1., "`random_rate` must be in [0, 1)"
        assert num_random_init > 0
        assert num_restarts > 0

        self.config_space = DenseConfigurationSpace(config_space, seed=seed)
        self.bounds = self.config_space.get_bounds()

        self.logit = self._build_compile_network(num_layers, num_units,
                                                 activation, optimizer)
        self.loss = self._build_loss(self.logit, normalize=normalize)

        self.gamma = gamma
        self.num_random_init = num_random_init
        self.random_rate = random_rate

        self.num_restarts = num_restarts
        self.method = method
        self.ftol = ftol
        self.max_iter = max_iter

        self.batch_size = batch_size
        self.num_steps_per_iter = num_steps_per_iter

        self.config_arrs = []
        self.losses = []

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def _array_from_dict(self, dct):
        config = DenseConfiguration(self.config_space, values=dct)
        return config.to_array()

    def _dict_from_array(self, array):
        config = DenseConfiguration.from_array(self.config_space,
                                               array_dense=array)
        return config.get_dictionary()

    def _get_dataset_size(self):
        return len(self.config_arrs)

    def _load_data(self):
        X = np.vstack(self.config_arrs)
        y = np.hstack(self.losses)
        return X, y

    def _load_labels(self, y):
        # TODO(LT): we can use clever data structures like heaps to make this
        #   labelling constant-time, but this is probably a premature
        #   optimization at this time...
        tau = np.quantile(y, q=self.gamma)
        return np.less(y, tau)

    def _get_steps_per_epoch(self, dataset_size):
        steps_per_epoch = int(np.ceil(np.true_divide(dataset_size,
                                                     self.batch_size)))
        return steps_per_epoch

    @staticmethod
    def _build_compile_network(num_layers, num_units, activation, optimizer):

        network = DenseSequential(output_dim=1,
                                  num_layers=num_layers,
                                  num_units=num_units,
                                  layer_kws=dict(activation=activation))
        network.compile(optimizer=optimizer, metrics=["accuracy"],
                        loss=BinaryCrossentropy(from_logits=True))
        return network

    @staticmethod
    def _build_loss(logit, normalize=True):

        if normalize:
            activation = tf.sigmoid
        else:
            activation = tf.identity

        @numpy_io
        @value_and_gradient
        @unbatch
        def loss(x):
            return - activation(logit(x))

        return loss

    def _update_model(self):

        X, y = self._load_data()
        z = self._load_labels(y)

        dataset_size = self._get_dataset_size()
        steps_per_epoch = self._get_steps_per_epoch(dataset_size)
        num_epochs = self.num_steps_per_iter // steps_per_epoch

        self.logit.fit(X, z, epochs=num_epochs, batch_size=self.batch_size,
                       verbose=False)  # TODO(LT): Make this an argument
        loss, accuracy = self.logit.evaluate(X, z, verbose=False)

        self.logger.info(f"[Model fit: loss={loss:.3f}, "
                         f"accuracy={accuracy:.3f}] "
                         f"dataset size: {dataset_size}, "
                         f"batch size: {self.batch_size}, "
                         f"steps per epoch: {steps_per_epoch}, "
                         f"num steps per iter: {self.num_steps_per_iter}, "
                         f"num epochs: {num_epochs}")

    def _get_maximum(self):

        self.logger.debug("Beginning multi-start maximization with "
                          f"{self.num_restarts} starts...")

        results = minimize_multi_start(self.loss, self.bounds,
                                       num_restarts=self.num_restarts,
                                       method=self.method, jac=True,
                                       options=dict(maxiter=self.max_iter,
                                                    ftol=self.ftol),
                                       random_state=self.random_state)

        res_best = None
        for i, res in enumerate(results):
            # TODO(LT): This currently assumes `normalize=False`
            self.logger.debug(f"[Maximum {i+1:02d}/{self.num_restarts:02d}: "
                              f"logit={-res.fun:.3f}] success: {res.success}, "
                              f"iterations: {res.nit:02d}, status: {res.status}"
                              f" ({res.message})")

            # TODO(LT): Create Enum type for these status codes
            if (res.status == 0 or res.status == 9) and \
                    not is_duplicate(res.x, self.config_arrs):
                # if (res_best is not None) *implies* (res.fun < res_best.fun)
                # (i.e. material implication) is logically equivalent to below
                if res_best is None or res.fun < res_best.fun:
                    res_best = res

        return res_best

    def get_config(self, budget):

        dataset_size = self._get_dataset_size()

        config_random = self.config_space.sample_configuration()
        config_random_dict = config_random.get_dictionary()

        if dataset_size < self.num_random_init:
            self.logger.debug(f"Completed {dataset_size}/{self.num_random_init}"
                              " initial runs. Returning random candidate...")
            return (config_random_dict, {})

        if self.random_state.binomial(p=self.random_rate, n=1):
            self.logger.info("[Glob. maximum: skipped "
                             f"(prob={self.random_rate:.2f})] "
                             "Returning random candidate ...")
            return (config_random_dict, {})

        # Update model
        self._update_model()

        # Maximize acquisition function
        opt = self._get_maximum()
        if opt is None:
            # TODO(LT): It's actually important to report what one of these
            # occurred...
            self.logger.warn("[Glob. maximum: not found!] Either optimization "
                             f"failed in all {self.num_restarts} starts, or "
                             "all maxima found have been evaluated previously!"
                             " Returning random candidate...")
            return (config_random_dict, {})

        # TODO(LT): This currently assumes `normalize=False`
        self.logger.info(f"[Glob. maximum: logit={-opt.fun:.3f}, "
                         f"prob={tf.sigmoid(-opt.fun):.3f}, "
                         f"rel. ratio={tf.sigmoid(-opt.fun)/self.gamma:.3f}] "
                         f"x={opt.x}")

        config_opt_arr = opt.x
        config_opt_dict = self._dict_from_array(config_opt_arr)

        return (config_opt_dict, {})

    def new_result(self, job, update_model=True):

        super(DRE, self).new_result(job)

        # TODO(LT): support multi-fidelity
        budget = job.kwargs["budget"]

        config_dict = job.kwargs["config"]
        config_arr = self._array_from_dict(config_dict)

        loss = job.result["loss"]

        self.config_arrs.append(config_arr)
        self.losses.append(loss)
