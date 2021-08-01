import numpy as np

from hpbandster.core.base_config_generator import base_config_generator
from hpbandster.optimizers.hyperband import HyperBand
from scipy.optimize import minimize
from tensorflow.keras.losses import BinaryCrossentropy

from ..decorators import numpy_io, squeeze, unbatch, value_and_gradient
from ..engine import Record, truncated_normal
from ..optimizers import multi_start, random_start
from ..types import DenseConfiguration, DenseConfigurationSpace

minimize_multi_start = multi_start(minimizer_fn=minimize)
minimize_random_start = random_start(minimizer_fn=minimize)


class BOREHyperband(HyperBand):

    def __init__(self, config_space, eta=3, min_budget=0.01, max_budget=1,
                 gamma=None, num_random_init=10, random_rate=0.1, retrain=False,
                 num_starts=5, num_samples=1024, batch_size=64,
                 num_steps_per_iter=1000, num_epochs=None, optimizer="adam",
                 num_layers=2, num_units=32, activation="elu", l2_factor=None,
                 transform="sigmoid", method="L-BFGS-B", max_iter=1000,
                 ftol=1e-9, distortion=None, seed=None, **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = SequenceClassifierConfigGenerator(config_space=config_space,
                                               gamma=gamma,
                                               num_random_init=num_random_init,
                                               random_rate=random_rate,
                                               retrain=retrain,
                                               classifier_kws=dict(
                                                num_layers=num_layers,
                                                num_units=num_units,
                                                l2_factor=l2_factor,
                                                activation=activation,
                                                optimizer=optimizer),
                                               fit_kws=dict(
                                                batch_size=batch_size,
                                                num_steps_per_iter=num_steps_per_iter,
                                                num_epochs=num_epochs),
                                               optimizer_kws=dict(
                                                transform=transform,
                                                method=method,
                                                max_iter=max_iter,
                                                ftol=ftol,
                                                distortion=distortion,
                                                num_starts=num_starts,
                                                num_samples=num_samples),
                                               seed=seed)
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


class SequenceClassifierConfigGenerator(base_config_generator):
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

        super(SequenceClassifierConfigGenerator, self).__init__(**kwargs)

        assert 0. < gamma < 1., "`gamma` must be in (0, 1)"
        assert num_random_init > 0
        assert random_rate is None or 0. <= random_rate < 1., \
            "`random_rate` must be in [0, 1)"

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

        self.factory = StackedRecurrentFactory(input_dim=input_dim,
                                               output_dim=1,
                                               num_layers=num_layers,
                                               num_units=num_units)

        # Build neural network probabilistic classifier
        self.logger.debug("Building and compiling network...")
        self.logit = self._build_compile_network(optimizer)
        self.logit.summary(print_fn=self.logger.debug)

        # Options for fitting neural network parameters
        self.batch_size = fit_kws["batch_size"]
        self.num_steps_per_iter = fit_kws["num_steps_per_iter"]

        # Options for maximizing the acquisition function
        self.final_activation = optimizer_kws["final_activation"]
        # The (negative) acquitions functions at various rungs.
        # TODO(LT): the name `losses` is not descriptive and a bit misleading.
        self.losses = {}

        assert optimizer_kws["num_start_points"] > 0
        self.num_start_points = optimizer_kws["num_start_points"]
        self.method = optimizer_kws["method"]
        self.ftol = optimizer_kws["ftol"]
        self.max_iter = optimizer_kws["max_iter"]
        self.distortion = optimizer_kws["distortion"]
        self.restart = optimizer_kws["restart"]

        self.record = Record(gamma=gamma)

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

    def _build_compile_network(self, optimizer):
        network = self.factory.build_many_to_many(mask_value=self.mask_value)
        network.compile(optimizer=optimizer, metrics=["accuracy"],
                        loss=BinaryCrossentropy(from_logits=True))
        return network

    def _build_loss(self, rung):
        """
        Returns the loss, i.e. the (negative) acquisition function to be
        minimized through the `scipy.optimize` interface.
        """
        num_steps = rung + 1  # rungs are zero-based
        classifier = self.factory.build_one_to_one(num_steps, activation=self.final_activation)
        # classifier.set_weights(weights=self.logit.get_weights())

        @numpy_io
        @value_and_gradient  # (D,) -> () to (D,) -> (), (D,)
        @squeeze(axis=-1)  # (D,) -> (1,) to (D,) -> ()
        @unbatch  # (None, D) -> (None, 1) to (D,) -> (1,)
        def loss(x):
            return - classifier(x)

        return loss

    def _update_model(self):

        inputs, targets = self.record.sequences_padded(binary=True, pad_value=self.mask_value)

        self.logger.debug(f"Input sequence shape: {inputs.shape}")
        self.logger.debug(f"Target sequence shape: {targets.shape}")

        num_epochs = 10
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

        # Negative acquisition functions specific to a rung. These are built
        # on-the-fly and then cached for subsequent use.
        loss = self.losses.setdefault(rung, self._build_loss(rung))

        # TODO(LT): There is a lot of redundant code duplicating and confusing
        # variable naming here.
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

        config_random = self.config_space.sample_configuration()
        config_random_dict = config_random.get_dictionary()

        # epsilon-greedy exploration
        if self.random_rate is not None and \
                self.random_state.binomial(p=self.random_rate, n=1):
            self.logger.info("[Glob. maximum: skipped "
                             f"(prob={self.random_rate:.2f})] "
                             "Suggesting random candidate ...")
            return (config_random_dict, {})

        # TODO(LT): Should just skip based on number of unique input features
        # observed so far.
        # Skip any model-based computation if there simply isn't enough
        # observed data yet
        highest = self.record.highest_rung(min_size=self.num_random_init)
        if highest is None:
            self.logger.debug("There are no rungs with at least "
                              f"{self.num_random_init} observations. "
                              "Suggesting random candidate...")
            return (config_random_dict, {})

        self.logger.debug(f"Rung {highest} is the highest with at least "
                          f"{self.num_random_init} observations.")

        # Update model
        self._update_model()

        # Maximize acquisition function
        opt = self._get_maximum(highest)
        if opt is None:
            # TODO(LT): It's actually important to report which of these
            # failures occurred...
            self.logger.warn("[Glob. maximum: not found!] Either optimization "
                             f"failed in all {self.num_start_points} starts, or "
                             "all maxima found have been evaluated previously!"
                             " Suggesting random candidate...")
            return (config_random_dict, {})

        loc = opt.x
        self.logger.info(f"[Glob. maximum: value={-opt.fun:.3f} x={loc}]")

        if self.distortion is None:
            config_opt_arr = loc
        else:
            dist = truncated_normal(loc=loc,
                                    scale=self.distortion,
                                    lower=self.bounds.lb,
                                    upper=self.bounds.ub)
            config_opt_arr = dist.rvs(random_state=self.random_state)
            self.logger.info(f"Suggesting x={config_opt_arr} "
                             f"(distortion={self.distortion:.3E})")

        config_opt_dict = self._dict_from_array(config_opt_arr)

        return (config_opt_dict, {})

    def new_result(self, job, update_model=True):

        super(SequenceClassifierConfigGenerator, self).new_result(job)

        # TODO(LT): support multi-fidelity
        budget = job.kwargs["budget"]

        config_dict = job.kwargs["config"]
        config_arr = self._array_from_dict(config_dict)

        loss = job.result["loss"]

        self.record.append(x=config_arr, y=loss, b=budget)

        self.logger.debug(f"[Data] rungs: {self.record.num_rungs()}, "
                          f"budgets: {self.record.budgets()}, "
                          f"rung sizes: {self.record.rung_sizes()}")
        self.logger.debug(f"[Data] thresholds: {self.record.thresholds()}")

        X = self.record.test()
        X_uniq, X_counts = np.unique(X, return_counts=True, axis=0)
        self.logger.debug(f"[Data] Input feature occurrences: {X_counts}")
        self.logger.debug(X_uniq)