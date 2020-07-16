import sys
import click
import json

import pandas as pd
import ConfigSpace
import hpbandster.core.nameserver as hpns

from hpbandster.optimizers.bohb import BOHB
from hpbandster.core.worker import Worker

from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                FCNetSliceLocalizationBenchmark,
                                FCNetNavalPropulsionBenchmark,
                                FCNetParkinsonsTelemonitoringBenchmark)
from pathlib import Path

OUTPUT_DIR = "results/"
INPUT_DIR = "datasets/fcnet_tabular_benchmarks"


def dataframe_from_result(result):

    rows = []

    for task, config_id in enumerate(result.data):

        d = result.data[config_id]
        bracket, _, _ = config_id

        for epoch in d.results:

            rows.append(dict(task=task,
                             bracket=bracket,
                             epoch=int(epoch),
                             error=d.results[epoch]["loss"],
                             cost=d.results[epoch]["info"],
                             submitted=d.time_stamps[epoch]["submitted"],
                             runtime=d.time_stamps[epoch]["finished"]))

    return pd.DataFrame(rows)


@click.command()
@click.argument("name")
@click.option("--benchmark-name", default="protein_structure")
@click.option("--input-dir", default=INPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, benchmark_name, input_dir, output_dir):

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO: Make these command-line arguments
    num_iterations = 10
    eta = 3
    num_samples = 64
    random_fraction = 1/3
    bandwidth_factor = 3
    min_bandwidth = 0.3
    min_budget = 3
    max_budget = 100

    if benchmark_name == "protein_structure":
        b = FCNetProteinStructureBenchmark(data_dir=input_dir)
    elif benchmark_name == "slice_localization":
        b = FCNetSliceLocalizationBenchmark(data_dir=input_dir)
    elif benchmark_name == "naval_propulsion":
        b = FCNetNavalPropulsionBenchmark(data_dir=input_dir)
    elif benchmark_name == "parkinsons_telemonitoring":
        b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=input_dir)
    else:
        # TODO: Raise some Exception
        pass

    cs = ConfigSpace.ConfigurationSpace()

    cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("n_units_1", lower=0, upper=5))
    cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("n_units_2", lower=0, upper=5))
    cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("dropout_1", lower=0, upper=2))
    cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("dropout_2", lower=0, upper=2))
    cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
    cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter("init_lr", lower=0, upper=5))
    cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
    cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("batch_size", lower=0, upper=3))

    class MyWorker(Worker):

        def compute(self, config, budget, **kwargs):

            original_cs = b.get_configuration_space()
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
            y, cost = b.objective_function(c, budget=int(budget))

            return dict(loss=float(y), info=float(cost))

    hb_run_id = '0'

    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()

    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id,
                     id=i)
        w.run(background=True)
        workers.append(w)

    bohb = BOHB(configspace=cs,
                run_id=hb_run_id,
                eta=eta,
                min_budget=min_budget,
                max_budget=max_budget,
                nameserver=ns_host,
                nameserver_port=ns_port,
                num_samples=num_samples,
                random_fraction=random_fraction,
                bandwidth_factor=bandwidth_factor,
                ping_interval=10,
                min_bandwidth=min_bandwidth)

    results = bohb.run(num_iterations, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    res = b.get_results()
    print(b.get_best_configuration())

    with open(output_path.joinpath("results.json"), 'w') as fh:
        json.dump(res, fh)

    df = dataframe_from_result(results)
    df.to_csv(output_path.joinpath("results.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover