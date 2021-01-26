import sys
import click
import yaml

import torch
import pandas as pd

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from botorch.utils.sampling import draw_sobol_samples

from pathlib import Path
from tqdm import trange
from datetime import datetime

from bore.types import DenseConfigurationSpace, DenseConfiguration
from bore.benchmarks import make_benchmark
from utils import make_name


def dict_from_tensor(tensor, cs):
    array = tensor.detach().cpu().numpy()
    config = DenseConfiguration.from_array(cs, array_dense=array)
    return config.get_dictionary()


def create_bounds(bounds, device=None, dtype=None):
    return torch.tensor([bounds.lb, bounds.ub], device=device, dtype=dtype)


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="gp-botorch")
@click.option("--num-runs", "-n", default=20)
@click.option("--run-start", default=0)
@click.option("--num-iterations", "-i", default=500)
@click.option("--acquisition-name", default="EI")
@click.option("--acquisition-optimizer-name", default="lbfgs",
              type=click.Choice(["lbfgs", "DIRECT", "CMA"]))
@click.option("--gamma", default=0.25, type=click.FloatRange(0., 1.),
              help="Quantile, or mixing proportion.")
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
         acquisition_optimizer_name, gamma, num_random_init, use_ard,
         use_input_warping, input_dir, output_dir):

    # TODO(LT): Turn into options
    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

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

    config_space = DenseConfigurationSpace(benchmark.get_config_space())
    bounds = create_bounds(config_space.get_bounds(), device=device, dtype=dtype)
    dim = config_space.get_dimensions()

    def objective(tensor, *args, **kwargs):
        """
        Wrapper that receives and returns torch.Tensor
        """
        config = dict_from_tensor(tensor, cs=config_space)
        res = - benchmark(config).value  # turn into maximization problem
        # TODO(LT): specify dtype and device
        return torch.tensor(res, requires_grad=False, device=device, dtype=dtype)

    # TODO(LT): make initial value option
    noise_variance = torch.tensor(1.0, device=device, dtype=dtype)

    for run_id in trange(run_start, num_runs, unit="run"):

        t_start = datetime.now()

        rows = []

        features = []
        targets = []

        with trange(num_iterations) as iterations:

            for i in iterations:

                if i < num_random_init:
                    # click.echo(f"Completed {i}/{num_random_init} initial runs. "
                    #            "Suggesting random candidate...")
                    # TODO(LT): specify dtype and device
                    # TODO(LT): support random seed
                    x_new = torch.rand(size=(dim,), device=device, dtype=dtype)
                else:
                    # construct dataset
                    X = torch.vstack(features)
                    y = torch.hstack(targets).unsqueeze(axis=-1)

                    # TODO(LT): persist model state, target standardization
                    # construct model
                    # model = FixedNoiseGP(X, standardize(y), noise_variance.expand_as(y),
                    model = FixedNoiseGP(X, y, noise_variance.expand_as(y),
                                         input_transform=None).to(X)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)

                    # update model
                    fit_gpytorch_model(mll)

                    # construct acquisition function
                    # TODO(LT): standardize?
                    tau = torch.quantile(y, q=1-gamma).item()
                    iterations.set_postfix(tau=tau)

                    ei = ExpectedImprovement(model=model, best_f=tau)

                    # optimize acquisition function
                    cand, _ = optimize_acqf(acq_function=ei, bounds=bounds, q=1,
                                            num_restarts=3, raw_samples=512,
                                            options=dict(batch_limit=5, maxiter=200))
                    x_new = cand.squeeze(axis=0)

                # evaluate blackbox objective
                y_new = objective(x_new)

                # update dataset
                features.append(x_new)
                targets.append(y_new)

                t = datetime.now()
                delta = t - t_start

                row = dict_from_tensor(x_new, cs=config_space)
                row["loss"] = - y_new.item()
                row["finished"] = delta.total_seconds()
                rows.append(row)

        data = pd.DataFrame(data=rows)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
