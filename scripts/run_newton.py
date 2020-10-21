import sys
import click
import yaml

import tensorflow as tf
import numpy as np

from scipy.optimize import basinhopping, minimize, Bounds
from bore.optimizers import random_start
from bore.decorators import value_and_gradient, numpy_io

from pathlib import Path


def log(x, f, accept):

    fg = "green" if accept else "red"
    click.secho(f"value {f:.3f} at {x}", fg=fg)


@numpy_io
@value_and_gradient
def michalewicz(x, m=10):

    N = x.shape[-1]
    n = tf.range(N, dtype="float64") + 1.

    a = tf.sin(x)
    b = tf.sin(n * x**2 / np.pi)
    b **= 2*m

    return - tf.reduce_sum(a * b, axis=-1)


@click.command()
@click.argument("dimensions", type=click.INT)
@click.option("--num-samples", "-n", default=50)
@click.option("--method", default="L-BFGS-B")
@click.option("--max-iter", "-m", default=10000)
@click.option("--ftol", default=1e-9)
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
@click.option('--seed', default=8888)
def main(dimensions, num_samples, method, max_iter, ftol, output_dir, seed):

    output_path = Path(output_dir).joinpath(f"michalewicz_{dimensions:03d}d", method)
    output_path.mkdir(parents=True, exist_ok=True)

    random_state = np.random.RandomState(seed)
    # minimize_multi_start = random_start(minimizer_fn=minimize)

    x_min, x_max = 0., np.pi
    lb = np.full(fill_value=x_min, shape=(dimensions,))
    ub = np.full(fill_value=x_max, shape=(dimensions,))
    bounds = Bounds(lb=lb, ub=ub)

    # best = minimize_multi_start(michalewicz, bounds,
    #                             num_samples=num_samples,
    #                             method=method, jac=True,
    #                             options=dict(maxiter=max_iter, ftol=ftol),
    #                             random_state=random_state)
    # best = min(filter(lambda res: res.success, results), key=lambda res: res.fun)

    # result = dict(X=list(map(float, best.x)), y=float(best.fun))
    # with output_path.joinpath("minimum.yaml").open("w") as f:
    #     yaml.dump(result, f)

    x0 = random_state.uniform(low=lb, high=ub)
    best = basinhopping(michalewicz, x0, niter=20, T=4.0,
                        minimizer_kwargs=dict(method="L-BFGS-B", jac=True,
                                              bounds=bounds, callback=print),
                        callback=log,
                        seed=random_state)
    print(best)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
