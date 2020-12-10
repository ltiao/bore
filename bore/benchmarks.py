import numpy as np
import ConfigSpace as CS
import yaml

from pathlib import Path
from collections import namedtuple
from abc import ABC, abstractmethod

from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                FCNetSliceLocalizationBenchmark,
                                FCNetNavalPropulsionBenchmark,
                                FCNetParkinsonsTelemonitoringBenchmark)
from .utils import config_space_to_search_space, kwargs_to_config

Evaluation = namedtuple('Evaluation', ['value', 'duration'])


class BenchmarkBase(ABC):

    @abstractmethod
    def get_config_space(self):
        pass

    @abstractmethod
    def get_minimum(self):
        pass


class Benchmark(BenchmarkBase):

    def get_search_space(self):
        cs = self.get_config_space()
        return config_space_to_search_space(cs)


class Forrester(Benchmark):

    def __call__(self, kwargs, budget=None):
        return Evaluation(value=self.func(**kwargs), duration=None)

    @staticmethod
    def func(x):
        return (6.*x - 2.)**2 * np.sin(12.*x-4.)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("x", lower=0., upper=1.))
        return cs

    def get_minimum(self):
        raise NotImplementedError
        # return 0.397887


class Sinusoid(Benchmark):

    def __call__(self, kwargs, budget=None):
        return Evaluation(value=self.func(**kwargs), duration=None)

    @staticmethod
    def func(x):
        return np.sin(3.0*x) + x**2 - 0.7*x

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("x", lower=-1.0, upper=2.0))
        return cs

    def get_minimum(self):
        raise NotImplementedError
        # return 0.397887


class Branin(Benchmark):

    def __init__(self, a=1.0, b=5.1/(4*np.pi**2), c=5.0/np.pi, r=6.0, s=10.0,
                 t=1.0/(8*np.pi)):
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t

    def __call__(self, kwargs, budget=None):

        value = self.func(**kwargs, a=self.a, b=self.b, c=self.c, r=self.r,
                          s=self.s, t=self.t)
        return Evaluation(value=value, duration=None)

    @staticmethod
    def func(x, y, a, b, c, r, s, t):
        return a*(y - b * x**2 + c*x - r)**2 + s*(1 - t)*np.cos(x) + s

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("x", lower=-5, upper=10))
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("y", lower=0, upper=15))
        return cs

    def get_minimum(self):
        return 0.397887


class Ackley(Benchmark):

    def __init__(self, dimensions, a=20, b=0.2, c=2*np.pi):
        self.dimensions = dimensions
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, kwargs, budget=None):
        x = np.hstack([kwargs.get(f"x{d}") for d in range(self.dimensions)])
        value = self.func(x, a=self.a, b=self.b, c=self.c)
        return Evaluation(value=value, duration=None)

    @staticmethod
    def func(x, a=20, b=0.2, c=2*np.pi):
        p = a * np.exp(-b * np.sqrt(np.mean(np.square(x), axis=-1)))
        q = np.exp(np.mean(np.cos(c*x)))
        return - p - q + a + np.e

    def get_minimum(self):
        return 0.

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for d in range(self.dimensions):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(f"x{d}", lower=-32.768, upper=32.768))
        return cs


class StyblinskiTang(Benchmark):

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def __call__(self, kwargs, budget=None):
        x = np.hstack([kwargs.get(f"x{d}") for d in range(self.dimensions)])
        return Evaluation(value=self.func(x), duration=None)

    @staticmethod
    def func(x):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5*x, axis=-1)

    def get_minimum(self):
        return -39.16599 * self.dimensions

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for d in range(self.dimensions):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(f"x{d}", lower=-5., upper=5.))
        return cs


class Michalewicz(Benchmark):

    def __init__(self, dimensions, m=10):
        self.dimensions = dimensions
        self.m = m

    def __call__(self, kwargs, budget=None):
        x = np.hstack([kwargs.get(f"x{d}") for d in range(self.dimensions)])
        return Evaluation(value=self.func(x, self.m), duration=None)

    @staticmethod
    def func(x, m):
        N = x.shape[-1]
        n = np.arange(N) + 1

        a = np.sin(x)
        b = np.sin(n * x**2 / np.pi)
        b **= 2*m
        return - np.sum(a * b, axis=-1)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for d in range(self.dimensions):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(f"x{d}", lower=0., upper=np.pi))
        return cs

    def get_minimum(self):
        minimums = {
            2: -1.8013,
            5: -4.687658,
            10: -9.66015
        }
        assert self.dimensions in minimums, \
            f"global minimum for dimensions={self.dimensions} not known"
        return minimums[self.dimensions]


class Hartmann(Benchmark):

    def __init__(self, dimensions, A, P, alpha=np.array([1.0, 1.2, 3.0, 3.2])):
        self.A = A
        self.P = P
        self.alpha = alpha
        self.dimensions = dimensions

    def __call__(self, kwargs, budget=None):
        x = np.hstack([kwargs.get(f"x{d}") for d in range(self.dimensions)])
        return Evaluation(value=self.func(x, self.A, self.P, self.alpha),
                          duration=None)

    @staticmethod
    def func(x, A, P, alpha):
        r = np.sum(A * np.square(x - P), axis=-1)
        return - np.dot(np.exp(-r), alpha)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for d in range(self.dimensions):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(f"x{d}", lower=0., upper=1.))
        return cs


class Hartmann3D(Hartmann):

    def __init__(self):
        A = np.array([[3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0],
                      [3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0]])
        P = 1e-4 * np.array([[3689, 1170, 2673],
                             [4699, 4387, 7470],
                             [1091, 8732, 5547],
                             [381,  5743, 8828]])
        super(Hartmann3D, self).__init__(dimensions=3, A=A, P=P)

    def get_minimum(self):
        return -3.86278


class Hartmann6D(Hartmann):

    def __init__(self):
        A = np.array([[10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
                      [0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
                      [3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
                      [17.0,  8.0,  0.05, 10.0,  0.1, 14.0]])
        P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091,  381]])
        super(Hartmann6D, self).__init__(dimensions=6, A=A, P=P)

    def get_minimum(self):
        return -3.32237


class FCNet(Benchmark):

    def __init__(self, dataset_name, data_dir):
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
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.benchmark = benchmark

    def __call__(self, kwargs, budget=100):
        cs = self.get_config_space()
        config = kwargs_to_config(kwargs, config_space=cs)
        y, cost = self.benchmark.objective_function(config)
        return Evaluation(value=y, duration=cost)

    def get_config_space(self):
        return self.benchmark.get_configuration_space()

    def get_minimum(self):

        base_path = Path(self.data_dir).joinpath(self.dataset_name)
        path = base_path.joinpath("minimum.yaml")

        if path.exists():
            with path.open('r') as f:
                val_error_min = yaml.safe_load(f).get("val_error_min")
        else:

            config_dict, val_error_min, \
                test_error_min = self.benchmark.get_best_configuration()

            d = dict(config_dict=config_dict,
                     val_error_min=float(val_error_min),
                     test_error_min=float(test_error_min))

            with path.open('w') as f:
                yaml.dump(d, f)
        return float(val_error_min)


class FCNetAlt(FCNet):

    def __call__(self, kwargs, budget=100):
        original_cs = self.benchmark.get_configuration_space()
        c = original_cs.sample_configuration()
        c["n_units_1"] = original_cs.get_hyperparameter("n_units_1").sequence[kwargs["n_units_1"]]
        c["n_units_2"] = original_cs.get_hyperparameter("n_units_2").sequence[kwargs["n_units_2"]]
        c["dropout_1"] = original_cs.get_hyperparameter("dropout_1").sequence[kwargs["dropout_1"]]
        c["dropout_2"] = original_cs.get_hyperparameter("dropout_2").sequence[kwargs["dropout_2"]]
        c["init_lr"] = original_cs.get_hyperparameter("init_lr").sequence[kwargs["init_lr"]]
        c["batch_size"] = original_cs.get_hyperparameter("batch_size").sequence[kwargs["batch_size"]]
        c["activation_fn_1"] = kwargs["activation_fn_1"]
        c["activation_fn_2"] = kwargs["activation_fn_2"]
        c["lr_schedule"] = kwargs["lr_schedule"]
        y, cost = self.benchmark.objective_function(c, budget=int(budget))
        return Evaluation(value=float(y), duration=float(cost))

    def get_config_space(self):
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


class GoldsteinPrice(Benchmark):

    def __call__(self, kwargs, budget=None):
        value = self.func(**kwargs)
        return Evaluation(value=value, duration=None)

    @staticmethod
    def func(x, y):
        a = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
        b = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
        return a * b

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("x", lower=-2., upper=2.))
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("y", lower=-2., upper=2.))
        return cs

    def get_minimum(self):
        return 3


class SixHumpCamel(Benchmark):

    def __call__(self, kwargs, budget=None):
        value = self.func(**kwargs)
        return Evaluation(value=value, duration=None)

    @staticmethod
    def func(x, y):
        return (4 - 2.1 * x**2 + x**4/3) * x**2 + x*y + (-4 + 4 * y**2) * y**2

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("x", lower=-3., upper=3.))
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter("y", lower=-2., upper=2.))
        return cs

    def get_minimum(self):
        return -1.0316

# def borehole(rw, r, Tu, Hu, Tl, Hl, L, Kw):

#     g = np.log(r) - np.log(rw)
#     h = 1.0 + 2.0 * L * Tu / (g * rw**2 * Kw) + Tu / Tl

#     ret = 2.0 * np.pi * Tu * (Hu - Hl)
#     ret /= g * h

#     return ret


# class BoreholeWorker(Worker):

#     def compute(self, config, budget, **kwargs):

#         y = - borehole(**config)

#         return dict(loss=y, info=None)

#     @staticmethod
#     def get_config_space():
#         cs = CS.ConfigurationSpace()
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("rw", lower=0.05, upper=0.15))
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("r", lower=100, upper=50000))
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("Tu", lower=63070, upper=115600))
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("Hu", lower=990, upper=1110))
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("Tl", lower=63.1, upper=116))
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("Hl", lower=700, upper=820))
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("L", lower=1120, upper=1680))
#         cs.add_hyperparameter(CS.UniformFloatHyperparameter("Kw", lower=9855, upper=12045))
#         return cs
# -309.5755876604079

benchmarks = dict(
    branin=Branin,
    goldstein_price=GoldsteinPrice,
    six_hump_camel=SixHumpCamel,
    styblinski_tang=StyblinskiTang,
    michalewicz=Michalewicz,
    hartmann3d=Hartmann3D,
    hartmann6d=Hartmann6D,
    fcnet=FCNet,
    fcnet_alt=FCNetAlt
)


def make_benchmark(benchmark_name, dimensions=None, dataset_name=None,
                   data_dir=None):

    Benchmark = benchmarks[benchmark_name]

    kws = {}
    if benchmark_name.startswith("fcnet"):
        assert dataset_name is not None, "must specify dataset name"
        assert data_dir is not None, "must specify data directory"
        kws["dataset_name"] = dataset_name
        kws["data_dir"] = data_dir

    if benchmark_name in ["michalewicz", "styblinski_tang"]:
        assert dimensions is not None, "must specify dimensions"
        kws["dimensions"] = dimensions

    return Benchmark(**kws)
