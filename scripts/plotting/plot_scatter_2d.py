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

from bore.benchmarks import branin, michalewicz, styblinski_tang
from utils import GOLDEN_RATIO, WIDTH, size, load_frame

OUTPUT_DIR = "figures/"


# def func(x, y):
#     return styblinski_tang(np.dstack([x, y]))

def func(x, y):
    return branin(x, y)


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
@click.option('--x-lim', type=(float, float), default=(0., np.pi))
@click.option('--y-lim', type=(float, float), default=(None, None))
@click.option('--x-num', default=512)
@click.option('--y-num', default=512)
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
def main(benchmark_name, method_name, num_runs, x_key, y_key, x_lim, y_lim,
         x_num, y_num, col_wrap, context, style, palette, width, aspect,
         extension, input_dir, output_dir):

    figsize = width_in, height_in = size(width, aspect)
    height = width / aspect
    suffix = f"{width:.0f}x{height:.0f}"

    x_min, x_max = x_lim
    y_min, y_max = y_lim
    if y_min is None or y_max is None:
        y_min, y_max = x_lim

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir).joinpath(benchmark_name)
    output_path = Path(output_dir).joinpath(benchmark_name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    y, x = np.ogrid[y_min:y_max:x_num * 1j, x_min:x_max:y_num * 1j]
    X, Y = np.broadcast_arrays(x, y)

    frames = []
    for run in trange(num_runs):

        path = input_path.joinpath(method_name, f"{run:03d}.csv")

        frame = load_frame(path, run)
        frames.append(frame.assign(method=method_name))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    # TODO: height should actually be `height_in / row_wrap`, but we don't
    # know what `row_wrap` is.
    g = sns.relplot(x=x_key, y=y_key, hue="evaluation", size="loss",
                    col="run", col_wrap=col_wrap,
                    palette="coolwarm", alpha=0.6,
                    height=height_in / col_wrap, aspect=aspect,
                    kind="scatter", data=data)
    g.map(contour, X=X, Y=Y, Z=func(X, Y), cmap="Spectral_r", zorder=-1)

    for ext in extension:
        g.savefig(output_path.joinpath(f"scatter_{context}_{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
