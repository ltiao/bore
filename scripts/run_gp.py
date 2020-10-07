import sys
import click
import yaml

import ConfigSpace as CS
import pandas as pd

import GPyOpt
from pathlib import Path

from bore.benchmarks import make_benchmark
from utils import make_name
from tqdm import trange


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="gp")
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
    config_space = benchmark.get_config_space()

    space = []
    for h in config_space.get_hyperparameters():

        if isinstance(h, CS.OrdinalHyperparameter):
            foo = dict(name=h.name, type="discrete",
                       domain=(0, len(h.sequence) - 1))
        elif isinstance(h, CS.CategoricalHyperparameter):
            foo = dict(name=h.name, type="categorical",
                       domain=[i for i, _ in enumerate(h.choices)])
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            foo = dict(name=h.name, type="discrete", domain=(h.lower, h.upper))
        elif isinstance(h, CS.UniformFloatHyperparameter):
            foo = dict(name=h.name, type="continuous", domain=(h.lower, h.upper),
                       dimensionality=1)
        space.append(foo)

    def func(x, *args, **kwargs):

        kws = {}
        for i, h in enumerate(config_space.get_hyperparameters()):
            if isinstance(h, CS.OrdinalHyperparameter):
                value = h.sequence[int(x[0, i])]
            elif isinstance(h, CS.CategoricalHyperparameter):
                value = h.choices[int(x[0, i])]
            elif isinstance(h, CS.UniformIntegerHyperparameter):
                value = int(x[0, i])
            else:
                value = x[0, i]
            kws[h.name] = value

        evaluation = benchmark(kws)
        return evaluation.value

    for run_id in trange(num_runs):

        BO = GPyOpt.methods.BayesianOptimization(f=func,
                                                 domain=space,
                                                 model_type="GP",
                                                 acquisition_type="EI",
                                                 normalize_Y=True,
                                                 exact_feval=False,
                                                 acquisition_optimizer_type="lbfgs")
        BO.run_optimization(num_iterations)

        data = pd.DataFrame(data=BO.X, columns=[foo["name"] for foo in space]) \
                 .assign(loss=BO.Y)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
