import numpy as np
import ConfigSpace

# from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2

from scipy.optimize import minimize, Bounds

from .models import DenseSequential
from .losses import binary_crossentropy_from_logits
from .decorators import unbatch, value_and_gradient, numpy_io

# from hpbandster.core.master import Master
from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.base_config_generator import base_config_generator


def get_bounds(config_space):

    lowers = []
    uppers = []

    for hp in config_space.get_hyperparameters():
        lowers.append(hp._inverse_transform(hp.lower))
        uppers.append(hp._inverse_transform(hp.upper))

    # return list(zip(lowers, uppers))
    return Bounds(lowers, uppers)


class BORE(HyperBand):

    def __init__(self, configspace, eta=3, min_budget=0.01, max_budget=1,
                 gamma=None, num_random_init=10, batch_size=64,
                 num_steps_per_iter=1000, optimizer="adam",
                 num_layers=2, num_units=32, activation="relu", seed=None,
                 **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = DRE(configspace=configspace,
                 gamma=gamma, num_random_init=num_random_init,
                 batch_size=batch_size, num_steps_per_iter=num_steps_per_iter,
                 optimizer=optimizer, num_layers=num_layers, num_units=num_units,
                 activation=activation, seed=seed)
        # (LT): Note this is using the *grandparent* class initializer to
        # replace the config_generator!
        super(HyperBand, self).__init__(config_generator=cg, **kwargs)

        # (LT): the following had to be copy-pasted as HpBandSter doesn't
        # really subscribe to the DRY design philosophy...

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
    def __init__(self, configspace, gamma=1/3, num_random_init=10,
                 batch_size=64, num_steps_per_iter=1000, optimizer="adam",
                 num_layers=2, num_units=32, activation="relu", seed=None,
                 **kwargs):

        super(DRE, self).__init__(**kwargs)

        self.configspace = configspace

        self.gamma = gamma
        self.num_random_init = num_random_init

        self.batch_size = batch_size
        self.num_steps_per_iter = num_steps_per_iter

        self.model = DenseSequential(output_dim=1,
                                     num_layers=num_layers,
                                     num_units=num_units,
                                     layer_kws=dict(activation=activation,
                                                    kernel_regularizer=l2(1e-4)))
        self.model.compile(optimizer=optimizer, metrics=["accuracy"],
                           loss=binary_crossentropy_from_logits)

        self.config_arrs = []
        self.losses = []

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def get_config(self, budget):

        dataset_size = len(self.config_arrs)

        if dataset_size < self.num_random_init:
            # TODO: how to seed this source of randomness?
            return (self.configspace.sample_configuration().get_dictionary(), {})

        @numpy_io()
        @value_and_gradient
        @unbatch
        def func(x):

            return - self.model(x)

        config_init = self.configspace.sample_configuration()
        config_init_arr = config_init.get_array()

        opt_res = minimize(func, x0=config_init_arr, jac=True,
                           bounds=get_bounds(self.configspace),
                           method="L-BFGS-B", tol=1e-8)
        config_opt_arr = opt_res.x
        config_opt = ConfigSpace.Configuration(self.configspace,
                                               vector=config_opt_arr)
        config_opt_dict = config_opt.get_dictionary()

        print(opt_res)
        return (config_opt_dict, {})

    def new_result(self, job, update_model=True):

        super(DRE, self).new_result(job)

        # TODO: ignoring this right now
        budget = job.kwargs["budget"]

        loss = job.result["loss"]
        config_dict = job.kwargs["config"]
        config = ConfigSpace.Configuration(self.configspace,
                                           values=config_dict)
        config_arr = config.get_array()

        self.losses.append(loss)
        self.config_arrs.append(config_arr)
        dataset_size = len(self.config_arrs)

        if dataset_size < self.num_random_init:
            self.logger.debug(f"Completed {dataset_size}/{self.num_random_init} "
                              "initial runs. Skipping model fitting.")
            return

        X = np.vstack(self.config_arrs)
        y = np.hstack(self.losses)

        y_threshold = np.quantile(y, q=self.gamma)
        z = np.less_equal(y, y_threshold)

        steps_per_epoch = int(np.ceil(np.true_divide(dataset_size,
                                                     self.batch_size)))
        num_epochs = self.num_steps_per_iter // steps_per_epoch

        print(f"Training model with {dataset_size} datapoints for "
              f"{num_epochs} epochs!")
        print(X, y)
        self.model.fit(X, z, epochs=num_epochs, batch_size=self.batch_size,
                       verbose=False)
