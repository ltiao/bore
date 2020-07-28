import sys
import click
import logging

# import numpy as np
# import pandas as pd

import hpbandster.core.nameserver as hpns

from pathlib import Path

from bore.engine import BORE
from bore.benchmarks import HartmannWorker
from bore.utils import dataframe_from_result

logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "results/"


@click.command()
@click.argument("name")
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, output_dir):

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO: Make these command-line arguments
    num_runs = 20
    num_iterations = 500

    gamma = 0.15
    num_random_init = 10
    num_restarts = 3
    batch_size = 64
    num_steps_per_iter = 100
    optimizer = "adam"
    num_layers = 2
    num_units = 32
    activation = "relu"

    eta = 3
    min_budget = 100
    max_budget = 100

    for run_id in range(num_runs):

        NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
        ns_host, ns_port = NS.start()

        num_workers = 1

        workers = []
        for i in range(num_workers):
            w = HartmannWorker(nameserver=ns_host, nameserver_port=ns_port,
                               run_id=run_id,
                               id=i)
            w.run(background=True)
            workers.append(w)

        rs = BORE(config_space=HartmannWorker.get_config_space(),
                  run_id=run_id,
                  eta=eta,
                  min_budget=min_budget,
                  max_budget=max_budget,
                  gamma=gamma,
                  num_random_init=num_random_init,
                  num_restarts=num_restarts,
                  batch_size=batch_size,
                  num_steps_per_iter=num_steps_per_iter,
                  optimizer=optimizer,
                  num_layers=num_layers,
                  num_units=num_units,
                  seed=run_id,
                  activation=activation,
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
