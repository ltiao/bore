import numpy as np

from tensorflow.keras.losses import BinaryCrossentropy

from ..engine import Ledger
from ..types import DenseConfigurationSpace, DenseConfiguration
from ..models import DenseSequential

from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.base_config_generator import base_config_generator


class BORE(HyperBand):

    def __init__(self, config_space, eta=3, min_budget=0.01, max_budget=1,
                 gamma=None, num_random_init=10, num_random_samples=64,
                 random_rate=0.1, batch_size=64, num_steps_per_iter=1000,
                 optimizer="adam", num_layers=2, num_units=32,
                 activation="relu", seed=None, **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = RatioEstimator(config_space=config_space, gamma=gamma,
                            num_random_init=num_random_init,
                            num_random_samples=num_random_samples,
                            random_rate=random_rate,
                            batch_size=batch_size,
                            num_steps_per_iter=num_steps_per_iter,
                            optimizer=optimizer, num_layers=num_layers,
                            num_units=num_units, activation=activation,
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


class RatioEstimator(base_config_generator):
    """
    class to implement random sampling from a ConfigSpace
    """
    def __init__(self, config_space, gamma=1/3, num_random_init=10,
                 num_random_samples=64, random_rate=0.1, batch_size=64,
                 num_steps_per_iter=1000, optimizer="adam", num_layers=2,
                 num_units=32, activation="relu", seed=None, **kwargs):

        super(RatioEstimator, self).__init__(**kwargs)

        assert 0. < gamma < 1., "`gamma` must be in (0, 1)"
        assert 0. <= random_rate < 1., "`random_rate` must be in [0, 1)"
        assert num_random_init > 0

        self.config_space = DenseConfigurationSpace(config_space, seed=seed)
        self.bounds = self.config_space.get_bounds()

        self.logit = self._build_compile_network(num_layers, num_units,
                                                 activation, optimizer)

        self.gamma = gamma
        self.num_random_init = num_random_init
        self.random_rate = random_rate

        self.num_random_samples = num_random_samples

        self.batch_size = batch_size
        self.num_steps_per_iter = num_steps_per_iter

        self.ledger = Ledger()

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
    def _build_compile_network(num_layers, num_units, activation, optimizer):

        network = DenseSequential(output_dim=1,
                                  num_layers=num_layers,
                                  num_units=num_units,
                                  layer_kws=dict(activation=activation))
        network.compile(optimizer=optimizer, metrics=["accuracy"],
                        loss=BinaryCrossentropy(from_logits=True))
        return network

    def _update_model(self):

        X, z = self.ledger.load_classification_data(self.gamma)

        dataset_size = self.ledger.size()
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

    def get_config(self, budget):

        dataset_size = self.ledger.size()

        config_random = next(self.config_space.sample_configuration())
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

        s = []
        for config_random in self.config_space.sample_configuration(size=self.num_random_samples):

            config_random_dict = config_random.get_dictionary()
            config_random_arr = self._array_from_dict(config_random_dict)
            s.append(config_random_arr)

        X_pred = np.vstack(s)
        y_pred = self.logit.predict(X_pred).squeeze(axis=-1)

        ind = y_pred.argmax()

        config_opt_dict = self._dict_from_array(s[ind])

        return (config_opt_dict, {})

    def new_result(self, job, update_model=True):

        super(RatioEstimator, self).new_result(job)

        # TODO(LT): support multi-fidelity
        budget = job.kwargs["budget"]

        config_dict = job.kwargs["config"]
        config_arr = self._array_from_dict(config_dict)

        loss = job.result["loss"]

        self.ledger.append(x=config_arr, y=loss, b=budget)
