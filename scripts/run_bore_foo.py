import sys
import click
import yaml
import logging

import hpbandster.core.nameserver as hpns

from pathlib import Path

from bore.plugins.hpbandster_foo import BORE
from bore.benchmarks import make_benchmark
from utils import make_name, BenchmarkWorker, HpBandSterLogs


logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="bore++")
@click.option("--num-runs", "-n", default=20)
@click.option("--run-start", default=0)
@click.option("--num-iterations", "-i", default=500)
@click.option("--eta", default=3, help="Successive halving reduction factor.")
@click.option("--min-budget", default=100)
@click.option("--max-budget", default=100)
@click.option("--gamma", default=1/3, type=click.FloatRange(0., 1.),
              help="Quantile, or mixing proportion.")
@click.option("--num-random-init", default=10)
@click.option("--num-epochs", default=200)
@click.option("--max-iter", default=20)
@click.option("--random-rate", default=0.1, type=click.FloatRange(0., 1.))
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations, eta, min_budget, max_budget, gamma,
         num_random_init, num_epochs, max_iter, random_rate, input_dir, output_dir):

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               data_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(eta=eta, min_budget=min_budget, max_budget=max_budget,
                   gamma=gamma, num_random_init=num_random_init,
                   num_epochs=num_epochs, max_iter=max_iter,
                   random_rate=random_rate)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    for run_id in range(run_start, num_runs):

        NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
        ns_host, ns_port = NS.start()

        num_workers = 1

        workers = []
        for worker_id in range(num_workers):
            w = BenchmarkWorker(benchmark=benchmark, nameserver=ns_host,
                                nameserver_port=ns_port, run_id=run_id,
                                id=worker_id)
            w.run(background=True)
            workers.append(w)

        rs = BORE(config_space=benchmark.get_config_space(),
                  run_id=run_id,
                  nameserver=ns_host,
                  nameserver_port=ns_port,
                  ping_interval=10,
                  seed=run_id,
                  **options)

        results = rs.run(num_iterations, min_n_workers=num_workers)

        rs.shutdown(shutdown_workers=True)
        NS.shutdown()

        data = HpBandSterLogs(results).to_frame()
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
