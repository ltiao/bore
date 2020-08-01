import sys
import click
# import json

# import numpy as np
# import pandas as pd

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

from pathlib import Path

from bore.benchmarks import (Hartmann3DWorker, Hartmann6DWorker,
                             BoreholeWorker, FCNetWorker, BraninWorker)
from bore.utils import dataframe_from_result


workers = dict(
    branin=BraninWorker,
    hartmann3d=Hartmann3DWorker,
    hartmann6d=Hartmann6DWorker,
    borehole=BoreholeWorker,
    fcnet=FCNetWorker
)


def get_worker(benchmark_name, dataset_name=None, input_dir=None):

    Worker = workers.get(benchmark_name)
    kws = {}

    if benchmark_name == "fcnet":
        assert dataset_name is not None, "must specify dataset name"
        kws["dataset_name"] = dataset_name
        kws["data_dir"] = input_dir

    return Worker, kws


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name")
@click.option("--method-name", default="random")
@click.option("--num-runs", "-n", default=20)
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, method_name, num_runs, input_dir,
         output_dir):

    Worker, worker_kws = get_worker(benchmark_name, dataset_name=dataset_name,
                                    input_dir=input_dir)

    # TODO: Make these command-line arguments
    num_iterations = 500

    eta = 3
    min_budget = 100
    max_budget = 100

    output_path = Path(output_dir).joinpath(f"{benchmark_name}_{dataset_name}",
                                            method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    for run_id in range(num_runs):

        NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
        ns_host, ns_port = NS.start()

        num_workers = 1

        workers = []
        for worker_id in range(num_workers):
            w = Worker(nameserver=ns_host, nameserver_port=ns_port,
                       run_id=run_id, id=worker_id, **worker_kws)
            w.run(background=True)
            workers.append(w)

        rs = RandomSearch(configspace=Worker.get_config_space(),
                          run_id=run_id,
                          eta=eta,
                          min_budget=min_budget,
                          max_budget=max_budget,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          ping_interval=10)

        results = rs.run(num_iterations, min_n_workers=num_workers)

        rs.shutdown(shutdown_workers=True)
        NS.shutdown()

        data = dataframe_from_result(results)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
