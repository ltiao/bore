import sys
import click
import yaml

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import BOHB

from pathlib import Path

from utils import make_name, make_benchmark, BenchmarkWorker, HpBandSterLogs


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="bohb")
@click.option("--num-runs", "-n", default=20)
@click.option("--num-iterations", "-i", default=500)
@click.option("--eta", default=3, help="Successive halving reduction factor.")
@click.option("--min-budget", default=100)
@click.option("--max-budget", default=100)
@click.option("--min-points-in-model", default=10)
@click.option("--top-n-percent", default=15)
@click.option("--num-samples", default=64)
@click.option("--random-fraction", default=1/3)
@click.option("--bandwidth-factor", default=3)
@click.option("--min-bandwidth", default=1e-3)
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         num_iterations, eta, min_budget, max_budget, min_points_in_model,
         top_n_percent, num_samples, random_fraction, bandwidth_factor,
         min_bandwidth, input_dir, output_dir):

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               input_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(eta=eta, min_budget=min_budget, max_budget=max_budget,
                   min_points_in_model=min_points_in_model,
                   top_n_percent=top_n_percent,
                   num_samples=num_samples,
                   random_fraction=random_fraction,
                   bandwidth_factor=bandwidth_factor,
                   min_bandwidth=min_bandwidth)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    for run_id in range(num_runs):

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

        rs = BOHB(configspace=benchmark.get_config_space(),
                  run_id=run_id,
                  nameserver=ns_host,
                  nameserver_port=ns_port,
                  ping_interval=10,
                  **options)

        results = rs.run(num_iterations, min_n_workers=num_workers)

        rs.shutdown(shutdown_workers=True)
        NS.shutdown()

        data = HpBandSterLogs(results).to_frame()
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover