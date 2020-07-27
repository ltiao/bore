import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from bore.utils import load_runs
from pathlib import Path

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


WIDTH = 397.48499
OUTPUT_DIR = "figures/"


@click.command()
@click.argument("name")
@click.argument("input_dir", default="results",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, input_dir, context, style, palette, width, aspect, extension,
         output_dir):

    figsize = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir)
    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    num_runs = 20
    error_min = -3.32237
    optimizers = ["random", "tpe", "borabora"]

    frames = []
    for optimizer in optimizers:

        frame = load_runs(input_path.joinpath(optimizer),
                          runs=num_runs, error_min=error_min)
        frames.append(frame.assign(optimizer=optimizer))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
    data.replace({"optimizer": {"random": "Random Search", "tpe": "TPE", "borabora": "BORE"}}, inplace=True)
    data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="task", y="regret best", hue="optimizer", style="optimizer",
                 # ci="sd",
                 data=data, ax=ax)

    ax.set_xlabel("iteration")
    ax.set_ylabel("incumbent regret")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5e-2, -error_min)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"regret_vs_iterations_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
