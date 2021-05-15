import numpy as np
import torch

from torch import nn
from torch.distributions.uniform import Uniform
from scipy.optimize import minimize

from ..stein.nn_classifier import NN
from ..stein.blr_classifier import BayesianLogisticRegression
from ..stein.stein import SVN, scaled_hessian_RBF

from ..stein.utils.LBFGS import FullBatchLBFGS

from ..data import Record
from ..types import DenseConfigurationSpace, DenseConfiguration

from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.base_config_generator import base_config_generator


class BORE(HyperBand):

    def __init__(self, config_space, eta=3, min_budget=0.01, max_budget=1,
                 gamma=None, num_random_init=10, num_epochs=200,
                 max_iter=20, random_rate=None, seed=None, **kwargs):

        if gamma is None:
            gamma = 1/eta

        cg = ClassifierGenerator(config_space=config_space, gamma=gamma,
                                 num_random_init=num_random_init,
                                 num_epochs=num_epochs, max_iter=max_iter,
                                 random_rate=random_rate,
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


class ClassifierGenerator(base_config_generator):
    """
    class to implement random sampling from a ConfigSpace
    """
    def __init__(self, config_space, gamma, num_random_init, num_epochs,
                 max_iter, random_rate, seed, **kwargs):

        super(ClassifierGenerator, self).__init__(**kwargs)

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

        bounds = self.config_space.get_bounds()
        self.constraints = torch.tensor([bounds.lb, bounds.ub],
                                        dtype=torch.float).transpose(0, 1)

        self.input_dim = self.config_space.get_dimensions(sparse=False)

        self.classifier = BayesianLogisticRegression(dim=self.input_dim)

        self.x_star = torch.rand([1, self.input_dim]).requires_grad_(True)
        self.optimiser = FullBatchLBFGS(params=[self.x_star], lr=1.,
                                        line_search='None')
        self.max_iter = max_iter

        self.num_epochs = num_epochs
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

    def _update_classifier(self):

        X, z = self.record.load_classification_data(self.gamma)

        X = torch.tensor(X, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)

        self.classifier.fit(X, z, nepoch=self.num_epochs, batch_size=None)

    def _get_maximum(self):
        # self.x_star = self.x_star.detach().requires_grad_(True)
        self.x_star = torch.rand([1, self.input_dim])

        # Scale to the correct range
        if self.constraints is not None:
            self.x_star[0, :] = self.constraints[:, 0] + self.x_star[0, :] * \
                                (self.constraints[:, 1] - self.constraints[:, 0])

        self.x_star = self.x_star.requires_grad_(True)
        # self.optimiser = FullBatchLBFGS(params=[self.x_star], lr=1., line_search='None')
        self.optimiser = torch.optim.RMSprop(params=[self.x_star], lr=.01)

        for i in range(self.max_iter):
            """
            Gradient-based optimisation
            """
            self.optimiser.zero_grad()
            loss = - self.classifier(self.x_star)
            loss.backward()
            self.optimiser.step()
            if self.constraints is not None:
                with torch.no_grad():
                    for j in range(self.input_dim):
                        self.x_star[0, j].clamp_(self.constraints[j, 0].item(),
                                                 self.constraints[j, 1].item())

        return self.x_star[0].detach().cpu().numpy()

    def get_config(self, budget):

        dataset_size = self.record.size()

        config_random = self.config_space.sample_configuration()
        config_random_dict = config_random.get_dictionary()

        # Insufficient training data
        if dataset_size < self.num_random_init:
            self.logger.debug(f"Completed {dataset_size}/{self.num_random_init}"
                              " initial runs. Suggesting random candidate...")
            return (config_random_dict, {})

        # Train classifier
        self._update_classifier()

        config_opt_arr = self._get_maximum()
        config_opt_dict = self._dict_from_array(config_opt_arr)

        return (config_opt_dict, {})

    def new_result(self, job, update_model=True):

        super(ClassifierGenerator, self).new_result(job)

        # TODO(LT): support multi-fidelity
        budget = job.kwargs["budget"]

        config_dict = job.kwargs["config"]
        config_arr = self._array_from_dict(config_dict)

        loss = job.result["loss"]

        self.record.append(x=config_arr, y=loss, b=budget)
