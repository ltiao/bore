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

    num_runs = 5
    # runs = list(range(20))
    # runs.pop(8)
    # runs.pop(13-1)

    # error_min = -3.32237
    error_min = -3.86278
    optimizers = ["random_hartmann3d", "tpe_hartmann3d", "test"]
    # optimizers = ["random", "tpe", "bore"]
    frames = []
    for optimizer in optimizers:

        frame = load_runs(input_path.joinpath(optimizer),
                          runs=num_runs, error_min=error_min)
        frames.append(frame.assign(optimizer=optimizer))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
    data.replace({"optimizer": {"random_hartmann3d": "Random Search",
                                "tpe_hartmann3d": "TPE",
                                "test": "BORE"}}, inplace=True)
    # data.replace({"optimizer": {"protein_structure": "Protein Structure"}}, inplace=True)
    data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="task", y="regret best", hue="optimizer", style="optimizer",
                 # units="run", estimator=None,
                 ci="sd", err_kws=dict(edgecolor='none'),
                 data=data, ax=ax)

    ax.set_xlabel("iteration")
    ax.set_ylabel("incumbent regret")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_ylim(5e-2, -error_min)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"regret_vs_iterations_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    g = sns.relplot(x="task", y="error", hue="epoch",
                    col="run", col_wrap=4, palette="Dark2",
                    alpha=0.6, kind="scatter", data=data.query("optimizer == 'BORE'"))
    g.map(plt.plot, "task", "best", color="k", linewidth=2.0, alpha=0.8)
    g.set_axis_labels("iteration", "regret")

    for ext in extension:
        g.savefig(output_path.joinpath(f"error_vs_iterations_{context}_{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
