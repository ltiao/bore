import numpy as np
import ConfigSpace as CS

from hpbandster.core.worker import Worker
from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                FCNetSliceLocalizationBenchmark,
                                FCNetNavalPropulsionBenchmark,
                                FCNetParkinsonsTelemonitoringBenchmark)


def hartmann(x, alpha, A, P):
    r = np.sum(A * np.square(x - P), axis=-1)
    return - np.dot(np.exp(-r), alpha)


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

        if dataset_name == "protein_structure":
            benchmark = FCNetProteinStructureBenchmark(data_dir=data_dir)
        elif dataset_name == "slice_localization":
            benchmark = FCNetSliceLocalizationBenchmark(data_dir=data_dir)
        elif dataset_name == "naval_propulsion":
            benchmark = FCNetNavalPropulsionBenchmark(data_dir=data_dir)
        elif dataset_name == "parkinsons_telemonitoring":
            benchmark = FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir)
        else:
            raise ValueError

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
