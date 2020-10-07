import sys
import click

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.colors import LogNorm

from tqdm import trange

from bore.benchmarks import Branin, Michalewicz, StyblinskiTang, GoldsteinPrice, SixHumpCamel
from utils import GOLDEN_RATIO, WIDTH, size, load_frame


OUTPUT_DIR = "figures/"


def contour(X, Y, Z, ax=None, *args, **kwargs):

    kwargs.pop("color")

    if ax is None:
        ax = plt.gca()

    ax.contour(X, Y, Z, *args, **kwargs)


@click.command()
@click.argument("benchmark_name")
@click.argument("method_name")
@click.option('--num-runs', '-n', default=20)
@click.option('--x-key', default='x0')
@click.option('--y-key', default='x1')
@click.option('--x-num', default=512)
@click.option('--y-num', default=512)
@click.option('--log-error-lim', type=(float, float), default=(-2, 3))
@click.option('--num-error-levels', default=20)
@click.option('--context', default="paper")
@click.option('--col-wrap', default=4)
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--input_dir", default="results",
              type=click.Path(file_okay=False, dir_okay=True))
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, method_name, num_runs, x_key, y_key, x_num, y_num,
         log_error_lim, num_error_levels, col_wrap, context, style, palette,
         width, aspect, extension, input_dir, output_dir):

    figsize = width_in, height_in = size(width, aspect)
    height = width / aspect
    suffix = f"{width:.0f}x{height:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir).joinpath(benchmark_name)
    output_path = Path(output_dir).joinpath(benchmark_name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    if benchmark_name == "branin":
        benchmark = Branin()
        def func(x, y):
            return benchmark(x, y) - benchmark.get_minimum()
    elif benchmark_name == "goldstein_price":
        benchmark = GoldsteinPrice()
        def func(x, y):
            return benchmark.func(x, y) - benchmark.get_minimum()
    elif benchmark_name == "six_hump_camel":
        benchmark = SixHumpCamel()
        def func(x, y):
            return benchmark.func(x, y) - benchmark.get_minimum()
    elif benchmark_name == "styblinski_tang_002d":
        benchmark = StyblinskiTang(dimensions=2)
        def func(x, y):
            return benchmark.func(np.dstack([x, y])) - benchmark.get_minimum()
    elif benchmark_name == "michalewicz_002d":
        benchmark = Michalewicz(dimensions=2)
        def func(x, y):
            return benchmark.func(np.dstack([x, y]), m=benchmark.m) - benchmark.get_minimum()

    cs = benchmark.get_config_space()
    hx = cs.get_hyperparameter(x_key)
    hy = cs.get_hyperparameter(y_key)

    y, x = np.ogrid[hy.lower:hy.upper:y_num * 1j, hx.lower:hx.upper:x_num * 1j]
    X, Y = np.broadcast_arrays(x, y)

    frames = []
    for run in trange(num_runs):

        path = input_path.joinpath(method_name, f"{run:03d}.csv")

        frame = load_frame(path, run, loss_min=benchmark.get_minimum())
        frames.append(frame.assign(method=method_name))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    # TODO: height should actually be `height_in / row_wrap`, but we don't
    # know what `row_wrap` is.
    g = sns.relplot(x=x_key, y=y_key, hue="evaluation", size="error",
                    col="run", col_wrap=col_wrap,
                    palette="cividis_r", alpha=0.8,
                    height=height_in / col_wrap, aspect=aspect,
                    kind="scatter", data=data)
    g.map(contour, X=X, Y=Y, Z=func(X, Y),
          levels=np.logspace(*log_error_lim, num_error_levels),
          norm=LogNorm(),
          alpha=0.4, cmap="turbo", zorder=-1)

    for ext in extension:
        g.savefig(output_path.joinpath(f"scatter_{context}_{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
