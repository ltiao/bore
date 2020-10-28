import numpy as np
import tensorflow as tf
from hpbandster.core.base_config_generator import base_config_generator
from hpbandster.optimizers.hyperband import HyperBand
from scipy.optimize import minimize
from scipy.stats import truncnorm
from tensorflow.keras.layers import (LSTM, Dense, Input, Masking, RepeatVector,
                                     TimeDistributed)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential

from ..decorators import numpy_io, squeeze, unbatch, value_and_gradient
from ..engine import Record
from ..models import DenseSequential
from ..optimizers import multi_start, random_start
from ..types import DenseConfiguration, DenseConfigurationSpace

minimize_multi_start = multi_start(minimizer_fn=minimize)
minimize_random_start = random_start(minimizer_fn=minimize)


def truncated_normal(loc, scale, lower, upper):
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm(a=a, b=b, loc=loc, scale=scale)


class StackedLSTMFactory:

    def __init__(self, input_dim, output_dim, num_layers=2, num_units=32,
                 layer_kws={}, final_layer_kws={}):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_layers = num_layers
        self.num_units = num_units

        assert "return_sequences" not in layer_kws
        assert "activation" not in final_layer_kws

        self.layer_kws = layer_kws
        self.final_layer_kws = final_layer_kws

    def build_network_foo(self, mask_value=1e+9):

        input_shape = (None, self.input_dim)

        network = Sequential()
        network.add(Masking(mask_value=mask_value, input_shape=input_shape))
        for _ in range(self.num_layers):
            network.add(LSTM(self.num_units, return_sequences=True,
                             **self.layer_kws))
        network.add(TimeDistributed(Dense(self.output_dim,
                                          **self.final_layer_kws)))
        return network

    def build_network_bar(self, num_steps, activation="sigmoid"):

        input_shape = (self.input_dim,)

        network = Sequential()
        network.add(RepeatVector(num_steps, input_shape=input_shape))
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)
            network.add(LSTM(self.num_units, return_sequences=return_sequences,
                             **self.layer_kws))
        network.add(Dense(self.output_dim, activation=activation,
                          **self.final_layer_kws))

        # if foo is not None:
        #     network.set_weights(weights=foo.get_weights())

        return network


class BORE(HyperBand):

    def __init__(self, config_space, eta=3, min_budget=0.01, max_budget=1,
                 gamma=None, num_random_init=10, random_rate=None,
                 num_start_points=10, batch_size=64, num_steps_per_iter=1000,
                 optimizer="adam", num_layers=2, num_units=32,
                 activation="relu", final_activation="sigmoid",
                 method="L-BFGS-B", max_iter=100, ftol=1e-2, distortion=None,
                 restart=False, seed=None, **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = RatioEstimator(config_space=config_space, gamma=gamma,
                            num_random_init=num_random_init,
                            random_rate=random_rate,
                            classifier_kws=dict(num_layers=num_layers,
                                                num_units=num_units,
                                                optimizer=optimizer,
                                                mask_value=1e+9),
                            fit_kws=dict(batch_size=batch_size,
                                         num_steps_per_iter=num_steps_per_iter),
                            optimizer_kws=dict(final_activation=final_activation,
                                               method=method,
                                               max_iter=max_iter,
                                               ftol=ftol,
                                               distortion=distortion,
                                               num_start_points=num_start_points,
                                               restart=restart), seed=seed)
        # (LT): Note this is using the *grandparent* class initializer to
        # replace the config_generator!
        super(HyperBand, self).__init__(config_generator=cg, **kwargs)

        # (LT): the design of HpBandSter framework requires us to copy-paste
        # the following boilerplate code (cannot really just subclass and
        # specify an alternative Configuration Generator).

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


class RatioEstimator(base_config_generator):
    """
    class to implement random sampling from a ConfigSpace
    """
    def __init__(self, config_space, gamma=1/3, num_random_init=10, random_rate=None,
                 classifier_kws=dict(num_layers=2, num_units=32,
                                     optimizer="adam", mask_value=1e+9),
                 fit_kws=dict(batch_size=64, num_steps_per_iter=100),
                 optimizer_kws=dict(final_activation="sigmoid",
                                    method="L-BFGS-B",
                                    max_iter=100,
                                    ftol=1e-2,
                                    distortion=None,
                                    num_start_points=3,
                                    restart=False),
                 seed=None, **kwargs):

        super(RatioEstimator, self).__init__(**kwargs)

        assert 0. < gamma < 1., "`gamma` must be in (0, 1)"
        assert num_random_init > 0
        assert random_rate is None or 0. <= random_rate < 1., \
            "`random_rate` must be in [0, 1)"

        self.gamma = gamma
        self.num_random_init = num_random_init
        self.random_rate = random_rate

        # Build ConfigSpace with one-hot-encoded categorical inputs and
        # initialize bounds
        self.config_space = DenseConfigurationSpace(config_space, seed=seed)
        self.bounds = self.config_space.get_bounds()

        input_dim = self.config_space.size_dense
        num_layers = classifier_kws["num_layers"]
        num_units = classifier_kws["num_units"]
        optimizer = classifier_kws["optimizer"]
        self.mask_value = classifier_kws["mask_value"]

        self.factory = StackedLSTMFactory(input_dim, output_dim=1,
                                          num_layers=num_layers,
                                          num_units=num_units)

        # Build neural network probabilistic classifier
        self.logger.debug("Building and compiling network...")
        self.logit = self.factory.build_network_foo(mask_value=self.mask_value)
        self.logit.compile(optimizer=optimizer, metrics=["accuracy"],
                           loss=BinaryCrossentropy(from_logits=True))
        self.logit.summary(print_fn=self.logger.debug)

        # Options for fitting neural network parameters
        self.batch_size = fit_kws["batch_size"]
        self.num_steps_per_iter = fit_kws["num_steps_per_iter"]

        # Options for maximizing the acquisition function
        self.final_activation = optimizer_kws["final_activation"]
        # self.loss = self._build_loss(activation=ACTIVATIONS[final_activation])

        assert optimizer_kws["num_start_points"] > 0
        self.num_start_points = optimizer_kws["num_start_points"]
        self.method = optimizer_kws["method"]
        self.ftol = optimizer_kws["ftol"]
        self.max_iter = optimizer_kws["max_iter"]
        self.distortion = optimizer_kws["distortion"]
        self.restart = optimizer_kws["restart"]

        self.record = Record()

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def _array_from_dict(self, dct):
        config = DenseConfiguration(self.config_space, values=dct)
        return config.to_array()

    def _dict_from_array(self, array):
        config = DenseConfiguration.from_array(self.config_space,
                                               array_dense=array)
        return config.get_dictionary()

    def _get_steps_per_epoch(self, dataset_size):
        steps_per_epoch = int(np.ceil(np.true_divide(dataset_size,
                                                     self.batch_size)))
        return steps_per_epoch

    # @staticmethod
    # def _build_compile_network(input_dims, optimizer, mask_value, **kwargs):

    #     network = Sequential([
    #         Masking(mask_value=mask_value, input_shape=(None, input_dims)),
    #         LSTM(units=32, return_sequences=True),
    #         TimeDistributed(Dense(1))
    #     ])
    #     network.compile(optimizer=optimizer, metrics=["accuracy"],
    #                     loss=BinaryCrossentropy(from_logits=True))
    #     return network

    def _build_loss(self, num_steps):
        """
        Returns the loss, i.e. the (negative) acquisition function to be
        minimized through the `scipy.optimize` interface.
        """
        classifier = self.factory.build_network_bar(num_steps, activation=(
                                                        self.final_activation))
        classifier.set_weights(weights=self.logit.get_weights())

        @numpy_io
        @value_and_gradient  # (D,) -> () to (D,) -> (), (D,)
        @squeeze(axis=-1)  # (D,) -> (1,) to (D,) -> ()
        @unbatch  # (None, D) -> (None, 1) to (D,) -> (1,)
        def loss(x):
            return - classifier(x)

        return loss

    def _update_model(self):

        inputs, targets = self.record.sequences_padded(gamma=self.gamma,
                                                       pad_value=self.mask_value)

        num_epochs = 100
        self.logit.fit(inputs, targets, epochs=num_epochs,
                       batch_size=self.batch_size, verbose=False)
        loss, accuracy = self.logit.evaluate(inputs, targets, verbose=False)

        self.logger.info(f"[Model fit: loss={loss:.3f}, "
                         f"accuracy={accuracy:.3f}] "
                         f"batch size: {self.batch_size}, "
                         f"num epochs: {num_epochs}")

    def _get_maximum(self, rung):

        self.logger.debug(f"Beginning multi-start maximization at rung {rung} "
                          f"with {self.num_start_points} starts...")

        loss = self._build_loss(rung)

        # TODO(LT): a lot of redundant code duplicating and confusing variable
        # naming here.
        if not self.restart:
            res = None
            i = 0
            while res is None or not (res.success or res.status == 1) \
                    or self.record.is_duplicate(res.x):
                res = minimize_random_start(loss, self.bounds,
                                            num_samples=self.num_start_points,
                                            method=self.method, jac=True,
                                            options=dict(maxiter=self.max_iter,
                                                         ftol=self.ftol),
                                            random_state=self.random_state)

                self.logger.debug(f"[Maximum {i+1:02d}: value={-res.fun:.3f}] "
                                  f"success: {res.success}, "
                                  f"iterations: {res.nit:02d}, "
                                  f"status: {res.status} ({res.message})")
                i += 1

            return res

        results = minimize_multi_start(loss, self.bounds,
                                       num_restarts=self.num_start_points,
                                       method=self.method, jac=True,
                                       options=dict(maxiter=self.max_iter,
                                                    ftol=self.ftol),
                                       random_state=self.random_state)

        res_best = None
        for i, res in enumerate(results):
            self.logger.debug(f"[Maximum {i+1:02d}/{self.num_start_points:02d}: "
                              f"value={-res.fun:.3f}] success: {res.success}, "
                              f"iterations: {res.nit:02d}, status: {res.status}"
                              f" ({res.message})")

            # TODO(LT): Create Enum type for these status codes
            # status == 1 signifies maximum iteration reached, which we don't
            # want to treat as a failure condition.
            if (res.success or res.status == 1) and \
                    not self.record.is_duplicate(res.x):
                if res_best is None or res.fun < res_best.fun:
                    res_best = res

        return res_best

    def get_config(self, budget):

        self.logger.warn(f"WHAT IS THIS BUDGET THING??? {budget}")

        config_random = self.config_space.sample_configuration()
        config_random_dict = config_random.get_dictionary()

        # Skip any model-based computation if there simply isn't enough
        # observed data yet
        rung = self.record.num_rungs(min_size=self.num_random_init)
        if not rung > 0:
            self.logger.debug("There are no rungs with at least "
                              f"{self.num_random_init} observations. "
                              "Suggesting random candidate...")
            return (config_random_dict, {})

        # epsilon-greedy exploration
        if self.random_rate is not None and \
                self.random_state.binomial(p=self.random_rate, n=1):
            self.logger.info("[Glob. maximum: skipped "
                             f"(prob={self.random_rate:.2f})] "
                             "Suggesting random candidate ...")
            return (config_random_dict, {})

        # Update model
        self._update_model()

        # Maximize acquisition function
        opt = self._get_maximum(rung)
        if opt is None:
            # TODO(LT): It's actually important to report what one of these
            # occurred...
            self.logger.warn("[Glob. maximum: not found!] Either optimization "
                             f"failed in all {self.num_start_points} starts, or "
                             "all maxima found have been evaluated previously!"
                             " Suggesting random candidate...")
            return (config_random_dict, {})

        loc = opt.x
        if self.distortion is None:
            config_opt_arr = loc
        else:
            dist = truncated_normal(loc=loc,
                                    scale=self.distortion,
                                    lower=self.bounds.lb,
                                    upper=self.bounds.ub)
            config_opt_arr = dist.rvs(random_state=self.random_state)

        self.logger.info(f"[Glob. maximum: value={-opt.fun:.3f} x={loc}] "
                         f"Suggesting x={config_opt_arr} "
                         f"(distortion={self.distortion:.3E})")

        config_opt_dict = self._dict_from_array(config_opt_arr)

        return (config_opt_dict, {})

    def new_result(self, job, update_model=True):

        super(RatioEstimator, self).new_result(job)

        # TODO(LT): support multi-fidelity
        budget = job.kwargs["budget"]

        config_dict = job.kwargs["config"]
        config_arr = self._array_from_dict(config_dict)

        loss = job.result["loss"]

        self.record.append(x=config_arr, y=loss, b=budget)

        self.logger.debug(f"[Data] rungs: {self.record.num_rungs()}, "
                          f"budgets: {self.record.budgets()}, "
                          f"rung sizes: {self.record.rung_sizes()}")
        self.logger.debug(f"[Data] largest rung with at least {self.num_random_init} "
                          f"observations: {self.record.num_rungs(min_size=self.num_random_init)}")
        self.logger.debug(f"[Data] thresholds: {self.record.thresholds(gamma=self.gamma)}")
