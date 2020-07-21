import numpy as np
import ConfigSpace

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform

from hpbandster.core.base_config_generator import base_config_generator
from scipy.optimize import minimize

from .models import DenseSequential
from .losses import binary_crossentropy_from_logits
from .decorators import negate, unbatch, value_and_gradient, numpy_io


class DRE(base_config_generator):
    """
    class to implement random sampling from a ConfigSpace
    """
    def __init__(self, config_space, quantile=1/3, num_random_init=10,
                 seed=None, **kwargs):
        """
        Parameters:
        -----------

        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
        **kwargs:
            see  hyperband.config_generators.base.base_config_generator for
            additional arguments
        """

        super(DRE, self).__init__(**kwargs)
        self.config_space = config_space

        self.config_arrs = []
        self.losses = []

        self.quantile = quantile
        self.num_random_init = 10

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

        # TODO: make these arguments
        num_layers = 1
        num_units = 8
        activation = "relu"
        optimizer = "adam"

        self.batch_size = 64
        self.num_steps = 100

        self.model = DenseSequential(output_dim=1,
                                     num_layers=num_layers,
                                     num_units=num_units,
                                     layer_kws=dict(activation=activation))
        self.model.compile(optimizer=optimizer, metrics=["accuracy"],
                           loss=binary_crossentropy_from_logits)

    def get_config(self, budget):

        # TODO: how to seed this source of randomness?
        return(self.config_space.sample_configuration().get_dictionary(), {})

    def new_result(self, job, update_model=True):

        super(DRE, self).new_result(job)

        # TODO: ignoring this right now
        budget = job.kwargs["budget"]

        loss = job.result["loss"]
        config_dict = job.kwargs["config"]
        config = ConfigSpace.Configuration(self.config_space, values=config_dict)
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

        y_threshold = np.quantile(y, q=self.quantile)
        z = np.less_equal(y, y_threshold)

        steps_per_epoch = int(np.ceil(np.true_divide(dataset_size, self.batch_size)))
        num_epochs = self.num_steps // steps_per_epoch

        print(f"Training model with {dataset_size} datapoints for {num_epochs} epochs!")
        self.model.fit(X, z, epochs=num_epochs, batch_size=self.batch_size)

# class Engine:

#     def __init__(self, optimizer="adam", seed=None):

#         self.xs = []
#         self.ys = []

#         self.model = Sequential()
#         self.model.add(Dense(32, activation="relu", kernel_initializer=GlorotUniform(seed=seed)))
#         self.model.add(Dense(32, activation="relu", kernel_initializer=GlorotUniform(seed=seed)))
#         self.model.add(Dense(1, kernel_initializer=GlorotUniform(seed=seed)))
#         self.model.compile(optimizer=optimizer, metrics=["accuracy"],
#                            loss=binary_crossentropy_from_logits)
#         self.seed = seed
#         self.random_state = np.random.RandomState(seed)

#     def fit(self):

#         X = np.vstack(self.xs)
#         y = np.hstack(self.ys)

#         return self.model.fit(X, y, epochs=2000, batch_size=100)

#     def update(self, X, y):

#         self.xs.append(X)
#         self.ys.append(y)

#     def get_maximum(self):

#         @numpy_io
#         @value_and_gradient
#         @unbatch
#         def func(x):
#             return - self.model(x)

#         x_init = self.random_state.uniform(low=xmin, high=xmax)
#         return minimize(func, x0=x_init, jac=True, method="L-BFGS-B", tol=1e-8)
