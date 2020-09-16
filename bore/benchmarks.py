import numpy as np
import ConfigSpace as CS

from hpbandster.core.worker import Worker
from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                FCNetSliceLocalizationBenchmark,
                                FCNetNavalPropulsionBenchmark,
                                FCNetParkinsonsTelemonitoringBenchmark)


def styblinski_tang(x):

    return 0.5 * np.sum(x**4 - 16 * x**2 + 5*x, axis=-1)


def michalewicz(x, m=10):

    N = x.shape[-1]
    n = np.arange(N) + 1

    a = np.sin(x)
    b = np.sin(n * x**2 / np.pi)
    b **= 2*m

    return - np.sum(a * b, axis=-1)


def goldstein_price(x, y):

    a = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    b = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 - 48*y + 36*x*y + 27*y**2)

    return a * b


def branin(x, y, a=1.0, b=5.1/(4*np.pi**2), c=5.0/np.pi, r=6.0, s=10.0,
           t=1.0/(8*np.pi)):
    return a*(y - b * x**2 + c*x - r)**2 + s*(1 - t)*np.cos(x) + s


def borehole(rw, r, Tu, Hu, Tl, Hl, L, Kw):

    g = np.log(r) - np.log(rw)
    h = 1.0 + 2.0 * L * Tu / (g * rw**2 * Kw) + Tu / Tl

    ret = 2.0 * np.pi * Tu * (Hu - Hl)
    ret /= g * h

    return ret


def hartmann(x, alpha, A, P):
    r = np.sum(A * np.square(x - P), axis=-1)
    return - np.dot(np.exp(-r), alpha)


class MichalewiczWorker(Worker):

    def __init__(self, dim, m=10, *args, **kwargs):

        super(MichalewiczWorker, self).__init__(*args, **kwargs)
        self.dim = dim
        self.m = m

    def compute(self, config, budget, **kwargs):

        X = np.hstack([config[f"x{d}"] for d in range(self.dim)])
        y = michalewicz(X, m=self.m)

        return dict(loss=y, info=None)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for d in range(self.dim):
            cs.add_hyperparameter(CS.UniformFloatHyperparameter(f"x{d}", lower=0., upper=np.pi))
        return cs


class StyblinskiTangWorker(Worker):

    def __init__(self, dim, *args, **kwargs):

        super(StyblinskiTangWorker, self).__init__(*args, **kwargs)
        self.dim = dim

    def compute(self, config, budget, **kwargs):

        X = np.hstack([config[f"x{d}"] for d in range(self.dim)])
        y = styblinski_tang(X)

        return dict(loss=y, info=None)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for d in range(self.dim):
            cs.add_hyperparameter(CS.UniformFloatHyperparameter(f"x{d}", lower=0., upper=np.pi))
        return cs


class BraninWorker(Worker):

    def compute(self, config, budget, **kwargs):
        y = branin(**config)
        return dict(loss=y, info=None)

    @staticmethod
    def get_config_space():
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("x", lower=-5, upper=10))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("y", lower=0, upper=15))
        return cs


class GoldsteinPriceWorker(Worker):

    def compute(self, config, budget, **kwargs):
        y = goldstein_price(**config)
        return dict(loss=y, info=None)

    @staticmethod
    def get_config_space():
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("x", lower=0, upper=1))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("y", lower=0, upper=1))
        return cs


class BoreholeWorker(Worker):

    def compute(self, config, budget, **kwargs):

        y = - borehole(**config)

        return dict(loss=y, info=None)

    @staticmethod
    def get_config_space():
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("rw", lower=0.05, upper=0.15))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("r", lower=100, upper=50000))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("Tu", lower=63070, upper=115600))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("Hu", lower=990, upper=1110))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("Tl", lower=63.1, upper=116))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("Hl", lower=700, upper=820))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("L", lower=1120, upper=1680))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("Kw", lower=9855, upper=12045))
        return cs


class HartmannWorker(Worker):

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def compute(self, config, budget, **kwargs):

        X = np.hstack([config[f"x{i}"] for i in range(self.dim)])
        y = hartmann(X, self.alpha, self.A, self.P)

        return dict(loss=y, info=None)

    @classmethod
    def get_config_space(cls):
        cs = CS.ConfigurationSpace()
        for i in range(cls.dim):
            cs.add_hyperparameter(CS.UniformFloatHyperparameter(
                f"x{i}", lower=0, upper=1))
        return cs


class Hartmann3DWorker(HartmannWorker):

    dim = 3
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381,  5743, 8828]])


class Hartmann6DWorker(HartmannWorker):

    dim = 6
    A = np.array([[10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
                  [0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
                  [3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
                  [17.0,  8.0,  0.05, 10.0,  0.1, 14.0]])
    P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091,  381]])


class FCNetWorker(Worker):

    def __init__(self, dataset_name, data_dir, *args, **kwargs):

        super(FCNetWorker, self).__init__(*args, **kwargs)

        if dataset_name == "protein":
            benchmark = FCNetProteinStructureBenchmark(data_dir=data_dir)
        elif dataset_name == "slice":
            benchmark = FCNetSliceLocalizationBenchmark(data_dir=data_dir)
        elif dataset_name == "naval":
            benchmark = FCNetNavalPropulsionBenchmark(data_dir=data_dir)
        elif dataset_name == "parkinsons":
            benchmark = FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir)
        else:
            raise ValueError("dataset name not recognized!")

        self.benchmark = benchmark

    def compute(self, config, budget, **kwargs):

        original_cs = self.benchmark.get_configuration_space()
        c = original_cs.sample_configuration()
        c["n_units_1"] = original_cs.get_hyperparameter("n_units_1").sequence[config["n_units_1"]]
        c["n_units_2"] = original_cs.get_hyperparameter("n_units_2").sequence[config["n_units_2"]]
        c["dropout_1"] = original_cs.get_hyperparameter("dropout_1").sequence[config["dropout_1"]]
        c["dropout_2"] = original_cs.get_hyperparameter("dropout_2").sequence[config["dropout_2"]]
        c["init_lr"] = original_cs.get_hyperparameter("init_lr").sequence[config["init_lr"]]
        c["batch_size"] = original_cs.get_hyperparameter("batch_size").sequence[config["batch_size"]]
        c["activation_fn_1"] = config["activation_fn_1"]
        c["activation_fn_2"] = config["activation_fn_2"]
        c["lr_schedule"] = config["lr_schedule"]
        y, cost = self.benchmark.objective_function(c, budget=int(budget))

        return dict(loss=float(y), info=float(cost))

    @staticmethod
    def get_config_space():
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("n_units_1", lower=0, upper=5))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("n_units_2", lower=0, upper=5))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("dropout_1", lower=0, upper=2))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("dropout_2", lower=0, upper=2))
        cs.add_hyperparameter(CS.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
        cs.add_hyperparameter(CS.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("init_lr", lower=0, upper=5))
        cs.add_hyperparameter(CS.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("batch_size", lower=0, upper=3))
        return cs
