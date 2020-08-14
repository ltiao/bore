import numpy as np
import tensorflow as tf

# from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2

from scipy.optimize import minimize

from .types import DenseConfigurationSpace, DenseConfiguration
from .models import DenseSequential
from .losses import binary_crossentropy_from_logits
from .decorators import unbatch, value_and_gradient, numpy_io
from .optimizers import multi_start

# from hpbandster.core.master import Master
from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.base_config_generator import base_config_generator


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
                 activation="relu", seed=None, **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = DRE(config_space=config_space,
                 gamma=gamma, num_random_init=num_random_init,
                 random_rate=random_rate, num_restarts=num_restarts,
                 batch_size=batch_size, num_steps_per_iter=num_steps_per_iter,
                 optimizer=optimizer, num_layers=num_layers,
                 num_units=num_units, activation=activation, seed=seed)
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
                 random_rate=0.25, num_restarts=10, batch_size=64,
                 num_steps_per_iter=1000, optimizer="adam", num_layers=2,
                 num_units=32, activation="relu", seed=None, **kwargs):

        super(DRE, self).__init__(**kwargs)

        self.config_space = DenseConfigurationSpace(config_space, seed=seed)

        self.gamma = gamma
        self.num_random_init = num_random_init

        assert 0. <= random_rate <= 1., "random rate must be in [0, 1]"
        self.random_rate = random_rate

        self.num_restarts = num_restarts

        self.batch_size = batch_size
        self.num_steps_per_iter = num_steps_per_iter

        self.config_arrs = []
        self.losses = []

        l2_factor = 1e-4

        self._init_model(num_layers, num_units, activation, optimizer, l2_factor)

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def _init_model(self, num_layers, num_units, activation, optimizer, l2_factor):

        self.model = DenseSequential(output_dim=1,
                                     num_layers=num_layers,
                                     num_units=num_units,
                                     layer_kws=dict(activation=activation,
                                                    kernel_regularizer=l2(l2_factor))) # TODO(LT): make this an argument
        self.model.compile(optimizer=optimizer, metrics=["accuracy"],
                           loss=binary_crossentropy_from_logits)

    @staticmethod
    def make_minimizer(num_restarts, method="L-BFGS-B", max_iter=10000,
                       tol=1e-8):

        @multi_start(num_restarts=num_restarts)
        def multi_start_minimizer(fn, x0, bounds):
            return minimize(fn, x0=x0, method=method, jac=True, bounds=bounds,
                            tol=tol, options=dict(maxiter=max_iter))

        return multi_start_minimizer

    def make_minimizee(self):

        @numpy_io
        @value_and_gradient
        @unbatch
        def func(x):

            return - tf.sigmoid(self.model(x))

        return func

    def get_config(self, budget):

        dataset_size = len(self.config_arrs)

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

        # Model fitting
        X = np.vstack(self.config_arrs)
        y = np.hstack(self.losses)

        y_threshold = np.quantile(y, q=self.gamma)
        z = np.less_equal(y, y_threshold)

        steps_per_epoch = int(np.ceil(np.true_divide(dataset_size,
                                                     self.batch_size)))
        num_epochs = self.num_steps_per_iter // steps_per_epoch

        self.model.fit(X, z, epochs=num_epochs, batch_size=self.batch_size,
                       verbose=False)  # TODO(LT): Make this an argument
        loss, accuracy = self.model.evaluate(X, z, verbose=False)

        self.logger.info(f"[Model fit: loss={loss:.3f}, "
                         f"accuracy={accuracy:.3f}] "
                         f"dataset size: {dataset_size}, "
                         f"batch size: {self.batch_size}, "
                         f"steps per epoch: {steps_per_epoch}, "
                         f"num steps per iter: {self.num_steps_per_iter}, "
                         f"num epochs: {num_epochs}")
        self.logger.debug(X)
        self.logger.debug(y)

        # Maximize acquisition function

        # TODO(LT): The following three assignments can all be done at
        #   initialization time
        minimize = self.make_minimizer(num_restarts=self.num_restarts)
        func = self.make_minimizee()
        bounds = self.config_space.get_bounds()

        self.logger.debug("Beginning multi-start maximization with "
                          f"{self.num_restarts} starts...")

        results = minimize(func, bounds, random_state=self.random_state)

        res_best = None
        for i, res in enumerate(results):
            self.logger.debug(f"[Maximum {i+1:02d}/{self.num_restarts:02d}: "
                              f"logit={-res.fun:.3f}] success: {res.success}, "
                              f"iterations: {res.nit:02d}, status: {res.status}"
                              f" ({res.message.decode('utf-8')})")

            if res.success and not is_duplicate(res.x, self.config_arrs):
                # if (res_best is not None) *implies* (res.fun < res_best.fun)
                # (i.e. material implication) is logically equivalent to below
                if res_best is None or res.fun < res_best.fun:
                    res_best = res

        if res_best is None:
            self.logger.warn("[Glob. maximum: not found!] Either optimization "
                             f"failed in all {self.num_restarts} starts, or "
                             "all maxima found have been evaluated previously!"
                             " Returning random candidate...")
            return (config_random_dict, {})

        self.logger.info(f"[Glob. maximum: logit={-res_best.fun:.3f}, "
                         f"prob={tf.sigmoid(-res_best.fun):.3f}, "
                         f"rel. ratio={tf.sigmoid(-res_best.fun)/self.gamma:.3f}] "
                         f"x={res_best.x}")

        config_opt_arr = res_best.x
        config_opt = DenseConfiguration.from_array(self.config_space,
                                                   array_dense=config_opt_arr)
        config_opt_dict = config_opt.get_dictionary()

        return (config_opt_dict, {})

    def new_result(self, job, update_model=True):

        super(DRE, self).new_result(job)

        # TODO: ignoring this right now
        budget = job.kwargs["budget"]

        loss = job.result["loss"]
        config_dict = job.kwargs["config"]
        config = DenseConfiguration(self.config_space, values=config_dict)
        config_arr = config.to_array()

        self.losses.append(loss)
        self.config_arrs.append(config_arr)
