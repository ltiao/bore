import yaml
import ConfigSpace as CS

from .base import Benchmark, Evaluation
from ..utils import kwargs_to_config

from pathlib import Path
from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                FCNetSliceLocalizationBenchmark,
                                FCNetNavalPropulsionBenchmark,
                                FCNetParkinsonsTelemonitoringBenchmark)


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
