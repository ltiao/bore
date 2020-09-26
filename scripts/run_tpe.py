import sys
import click
import json

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from bore.benchmarks import branin
from pathlib import Path

from utils import get_worker, get_name, HyperOptLogs


def objective(kws):

    return dict(loss=branin(**kws), status=STATUS_OK, info=5)


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="tpe")
@click.option("--num-runs", "-n", default=20)
@click.option("--num-iterations", "-i", default=500)
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         num_iterations, input_dir, output_dir):

    Worker, worker_kws = get_worker(benchmark_name, dimensions=dimensions,
                                    dataset_name=dataset_name,
                                    input_dir=input_dir)
    name = get_name(benchmark_name, dimensions=dimensions, dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(num_iterations=num_iterations)
    with open(output_path.joinpath("options.json"), 'w') as f:
        json.dump(options, f, sort_keys=True, indent=2)

    space = {
        'x': hp.uniform('x', -5, 10),
        'y': hp.uniform('y', 0, 15)
    }

    for run_id in range(num_runs):

        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest,
                    max_evals=num_iterations, trials=trials)

        data = HyperOptLogs(trials).to_frame()
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
