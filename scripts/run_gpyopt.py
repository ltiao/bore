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


def dict_from_array(array, cs):

    kws = {}
    for i, h in enumerate(cs.get_hyperparameters()):
        if isinstance(h, CS.OrdinalHyperparameter):
            value = h.sequence[int(array[0, i])]
        elif isinstance(h, CS.CategoricalHyperparameter):
            value = h.choices[int(array[0, i])]
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            value = int(array[0, i])
        else:
            value = array[0, i]
        kws[h.name] = value
    return kws


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="gp")
@click.option("--num-runs", "-n", default=20)
@click.option("--run-start", default=0)
@click.option("--num-iterations", "-i", default=500)
@click.option("--acquisition-name", default="EI")
@click.option("--acquisition-optimizer-name", default="lbfgs",
              type=click.Choice(["lbfgs", "DIRECT", "CMA"]))
@click.option("--num-random-init", default=10)
@click.option('--use-ard', is_flag=True)
@click.option('--use-input-warping', is_flag=True)
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations, acquisition_name,
         acquisition_optimizer_name, num_random_init, use_ard,
         use_input_warping, input_dir, output_dir):

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               data_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(acquisition_name=acquisition_name,
                   acquisition_optimizer_name=acquisition_optimizer_name,
                   use_ard=use_ard, use_input_warping=use_input_warping)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    space = benchmark.get_domain()
    config_space = benchmark.get_config_space()

    def func(array, *args, **kwargs):
        kws = dict_from_array(array, cs=config_space)
        return benchmark(kws).value

    model_type = "input_warped_GP" if use_input_warping else "GP"

    for run_id in trange(run_start, num_runs):

        BO = GPyOpt.methods.BayesianOptimization(f=func, domain=space,
                                                 initial_design_numdata=num_random_init,
                                                 model_type=model_type,
                                                 ARD=use_ard,
                                                 normalize_Y=True,
                                                 exact_feval=False,
                                                 acquisition_type=acquisition_name,
                                                 acquisition_optimizer_type=acquisition_optimizer_name)
        BO.run_optimization(num_iterations)

        data = pd.DataFrame(data=BO.X, columns=[d["name"] for d in space]) \
                 .assign(loss=BO.Y)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
