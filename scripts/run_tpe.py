import sys
import click
import yaml

from hyperopt import fmin, tpe, STATUS_OK, Trials
from pathlib import Path

from utils import make_name, make_benchmark, HyperOptLogs


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

    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict()
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               input_dir=input_dir)
    space = benchmark.get_search_space()

    def objective(kws):
        evaluation = benchmark(kws)
        return dict(loss=evaluation.value, status=STATUS_OK,
                    info=evaluation.duration)

    for run_id in range(num_runs):

        trials = Trials()
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_iterations,
                    trials=trials)

        data = HyperOptLogs(trials).to_frame()
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
