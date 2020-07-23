import sys
import click
# import json

# import numpy as np
# import pandas as pd

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import BOHB

from pathlib import Path

from bore.benchmarks import HartmannWorker
from bore.utils import dataframe_from_result

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
    num_iterations = 200

    min_points_in_model = 10
    top_n_percent = 15
    num_samples = 64
    random_fraction = 1/3
    bandwidth_factor = 3
    min_bandwidth = 0.3

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

        rs = BOHB(configspace=HartmannWorker.get_configspace(),
                  run_id=run_id,
                  eta=eta,
                  min_budget=min_budget,
                  max_budget=max_budget,
                  min_points_in_model=min_points_in_model,
                  top_n_percent=top_n_percent,
                  num_samples=num_samples,
                  random_fraction=random_fraction,
                  bandwidth_factor=bandwidth_factor,
                  min_bandwidth=min_bandwidth,
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
