import sys
import click
import json
import logging

import hpbandster.core.nameserver as hpns

from pathlib import Path

from bore.plugins.hpbandster import BORE
from bore.utils import dataframe_from_result
from utils import get_worker, get_name


logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="bore")
@click.option("--num-runs", "-n", default=20)
@click.option("--num-iterations", "-i", default=500)
@click.option("--eta", default=3, help="Successive halving reduction factor.")
@click.option("--min-budget", default=100)
@click.option("--max-budget", default=100)
@click.option("--gamma", default=1/3, type=click.FloatRange(0., 1.),
              help="Quantile, or mixing proportion.")
@click.option("--num-random-init", default=10)
@click.option("--num-random-samples", default=64)
@click.option("--random-rate", default=0.1, type=click.FloatRange(0., 1.))
@click.option("--batch-size", default=64)
@click.option("--num-steps-per-iter", default=100)
@click.option("--optimizer", default="adam")
@click.option("--num-layers", default=2)
@click.option("--num-units", default=32)
@click.option("--activation", default="elu")
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         num_iterations, eta, min_budget, max_budget, gamma, num_random_init,
         num_random_samples, random_rate, batch_size, num_steps_per_iter,
         optimizer, num_layers, num_units, activation, input_dir, output_dir):

    Worker, worker_kws = get_worker(benchmark_name, dimensions=dimensions,
                                    dataset_name=dataset_name,
                                    input_dir=input_dir)
    name = get_name(benchmark_name, dimensions=dimensions, dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(num_iterations=num_iterations,
                   eta=eta, min_budget=min_budget, max_budget=max_budget,
                   gamma=gamma, num_random_init=num_random_init,
                   num_random_samples=num_random_samples, random_rate=random_rate,
                   batch_size=batch_size, num_steps_per_iter=num_steps_per_iter,
                   optimizer=optimizer, num_layers=num_layers,
                   num_units=num_units, activation=activation)

    with open(output_path.joinpath("options.json"), 'w') as f:
        json.dump(options, f, sort_keys=True, indent=2)

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

        rs = BORE(config_space=w.get_config_space(),
                  run_id=run_id,
                  eta=eta,
                  min_budget=min_budget,
                  max_budget=max_budget,
                  gamma=gamma,
                  num_random_init=num_random_init,
                  num_random_samples=num_random_samples,
                  random_rate=random_rate,
                  batch_size=batch_size,
                  num_steps_per_iter=num_steps_per_iter,
                  optimizer=optimizer,
                  num_layers=num_layers,
                  num_units=num_units,
                  activation=activation,
                  seed=run_id,
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
