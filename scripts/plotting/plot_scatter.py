import sys
import click

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.colors import LogNorm

from tqdm import tqdm, trange
from itertools import product

from bore.benchmarks import branin
from utils import GOLDEN_RATIO, WIDTH, size

OUTPUT_DIR = "figures/"


@click.command()
@click.argument("benchmark_name")
@click.argument("input_dir", default="results",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--vars', '-v', multiple=True)
@click.option('--num-runs', default=20)
@click.option('--context', default="paper")
@click.option('--col-wrap', default=3)
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, input_dir, vars, num_runs, col_wrap, context, style,
         palette, width, aspect, extension, output_dir):

    figsize = size(width, aspect)
    height = width / aspect
    suffix = f"{width:.0f}x{height:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir).joinpath(benchmark_name)
    output_path = Path(output_dir).joinpath(benchmark_name)
    output_path.mkdir(parents=True, exist_ok=True)

    for a, b, c, d in tqdm(product(["logit", "sigmoid"], ["elu", "relu"],
                                   ["1e-9", "1e-2"], ["0.15", "0.33333333333333333333"])):

        method = f"bore-{a}-{b}-ftol-{c}-gamma-{d}"
        for run in trange(num_runs):
            path = input_path.joinpath(method, f"{run:03d}.csv")
            frame = pd.read_csv(path, index_col=0)

            g = sns.pairplot(frame, vars=vars,  # diag_kind="kde",
                             height=height, aspect=aspect,
                             corner=True)
            for ext in extension:
                g.savefig(output_path.joinpath(f"scatter_{a}_{b}_"
                                               f"ftol_{c}_gamma_{d}_"
                                               f"{run:03d}_{context}_"
                                               f"{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
