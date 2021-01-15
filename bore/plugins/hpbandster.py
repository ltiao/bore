import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.losses import BinaryCrossentropy
from scipy.optimize import minimize

from ..data import Record
from ..engine import convert, truncated_normal
from ..models import DenseMaximizableSequential
from ..optimizers import multi_start, random_start
from ..types import DenseConfigurationSpace, DenseConfiguration

from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.base_config_generator import base_config_generator


TRANSFORMS = dict(identity=tf.identity, sigmoid=tf.sigmoid, exp=tf.exp)

minimize_multi_start = multi_start(minimizer_fn=minimize)
minimize_random_start = random_start(minimizer_fn=minimize)


class BORE(HyperBand):

    def __init__(self, config_space, eta=3, min_budget=0.01, max_budget=1,
                 gamma=None, num_random_init=10, random_rate=None, retrain=False,
                 num_start_points=10, batch_size=64, num_steps_per_iter=1000,
                 num_epochs=None, optimizer="adam", num_layers=2, num_units=32,
                 activation="relu", transform="sigmoid",
                 method="L-BFGS-B", max_iter=100, ftol=1e-2, distortion=None,
                 restart=False, seed=None, **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = RatioEstimator(config_space=config_space, gamma=gamma,
                            num_random_init=num_random_init,
                            random_rate=random_rate, retrain=retrain,
                            classifier_kws=dict(num_layers=num_layers,
                                                num_units=num_units,
                                                activation=activation,
                                                optimizer=optimizer),
                            fit_kws=dict(batch_size=batch_size,
                                         num_steps_per_iter=num_steps_per_iter,
                                         num_epochs=num_epochs),
                            optimizer_kws=dict(transform=transform,
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
                 retrain=False,
                 classifier_kws=dict(num_layers=2,
                                     num_units=32,
                                     activation="relu",
                                     optimizer="adam"),
                 fit_kws=dict(batch_size=64, num_epochs=None, num_steps_per_iter=100),
                 optimizer_kws=dict(transform="sigmoid",
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

        self.input_dim = input_dim
        self.classifier_kws = classifier_kws

        self.logit = None
        if not retrain:
            # Build neural network probabilistic classifier
            self.logger.debug("Building and compiling network...")
            self.logit = self._build_compile_network(input_dim, **classifier_kws)
            self.logit.summary(print_fn=self.logger.debug)

        # Options for fitting neural network parameters
        self.batch_size = fit_kws["batch_size"]
        self.num_steps_per_iter = fit_kws["num_steps_per_iter"]
        self.num_epochs = fit_kws["num_epochs"]

        # Options for maximizing the acquisition function
        transform_name = optimizer_kws["transform"]
        assert transform_name in TRANSFORMS, \
            f"`transform` must be one of {tuple(TRANSFORMS.keys())}"
        self.transform = TRANSFORMS[transform_name]
        self.loss = None
        if not retrain:
            self.loss = convert(self.logit, transform=lambda u: - self.transform(u))

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

    @staticmethod
    def _build_compile_network(input_dim, num_layers, num_units, activation, optimizer):

        network = DenseMaximizableSequential(input_dim=input_dim, output_dim=1,
                                             num_layers=num_layers,
                                             num_units=num_units,
                                             layer_kws=dict(
                                                activation=activation))
        network.compile(optimizer=optimizer, metrics=["accuracy"],
                        loss=BinaryCrossentropy(from_logits=True))
        return network

    def _update_model(self, logit):

        X, z = self.record.load_classification_data(self.gamma)

        dataset_size = self.record.size()
        steps_per_epoch = self._get_steps_per_epoch(dataset_size)

        num_epochs = self.num_epochs
        if num_epochs is None:
            num_epochs = self.num_steps_per_iter // steps_per_epoch
            self.logger.debug("Argument `num_epochs` has not been specified. "
                              f"Setting num_epochs={num_epochs}")
        else:
            self.logger.debug("Argument `num_epochs` is specified "
                              f"(num_epochs={num_epochs}). "
                              f"Ignoring num_steps_per_iter={self.num_steps_per_iter}")

        logit.fit(X, z, epochs=num_epochs, batch_size=self.batch_size,
                  verbose=False)  # TODO(LT): Make this an argument
        loss, accuracy = logit.evaluate(X, z, verbose=False)

        self.logger.info(f"[Model fit: loss={loss:.3f}, "
                         f"accuracy={accuracy:.3f}] "
                         f"dataset size: {dataset_size}, "
                         f"batch size: {self.batch_size}, "
                         f"steps per epoch: {steps_per_epoch}, "
                         f"num steps per iter: {self.num_steps_per_iter}, "
                         f"num epochs: {num_epochs}")

    def _get_maximum(self, logit):

        self.logger.debug("Beginning multi-start maximization with "
                          f"{self.num_start_points} starts...")

        loss = self.loss
        if loss is None:
            loss = convert(logit, transform=lambda u: - self.transform(u))

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

        dataset_size = self.record.size()

        config_random = self.config_space.sample_configuration()
        config_random_dict = config_random.get_dictionary()

        # epsilon-greedy exploration
        if self.random_rate is not None and \
                self.random_state.binomial(p=self.random_rate, n=1):
            self.logger.info("[Glob. maximum: skipped "
                             f"(prob={self.random_rate:.2f})] "
                             "Suggesting random candidate ...")
            return (config_random_dict, {})

        # Insufficient training data
        if dataset_size < self.num_random_init:
            self.logger.debug(f"Completed {dataset_size}/{self.num_random_init}"
                              " initial runs. Suggesting random candidate...")
            return (config_random_dict, {})

        logit = self.logit
        delete = False
        if logit is None:
            # Build neural network probabilistic classifier
            self.logger.debug("Building and compiling network...")
            logit = self._build_compile_network(self.input_dim, **self.classifier_kws)
            logit.summary(print_fn=self.logger.debug)
            delete = True  # mark for deletion at the end of iteration

        # Update model
        self._update_model(logit=logit)

        # Maximize acquisition function
        opt = self._get_maximum(logit)
        if opt is None:
            # TODO(LT): It's actually important to report which of these
            # failures occurred...
            self.logger.warn("[Glob. maximum: not found!] Either optimization "
                             f"failed in all {self.num_start_points} starts, or "
                             "all maxima found have been evaluated previously!"
                             " Suggesting random candidate...")
            return (config_random_dict, {})

        if delete:
            # if we are not persisting model across BO iterations
            # delete and clear from memory
            self.logger.debug("Deleting model...")
            K.clear_session()
            del logit

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

        super(RatioEstimator, self).new_result(job)

        # TODO(LT): support multi-fidelity
        budget = job.kwargs["budget"]

        config_dict = job.kwargs["config"]
        config_arr = self._array_from_dict(config_dict)

        loss = job.result["loss"]

        self.record.append(x=config_arr, y=loss, b=budget)
