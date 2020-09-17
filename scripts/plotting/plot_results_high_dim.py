import sys
import click

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from utils import GOLDEN_RATIO, WIDTH, size, load_frame, get_error_mins

OUTPUT_DIR = "figures/"


@click.command()
@click.argument("base_benchmark_name")
@click.argument("input_dir", default="results",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--dimensions', '-d', multiple=True, type=int)
@click.option('--num-runs', '-n', default=20)
@click.option('--methods', '-m', multiple=True)
@click.option('--ci')
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(base_benchmark_name, input_dir, dimensions, num_runs, methods, ci,
         context, style, palette, width, aspect, extension, output_dir):

    figsize = size(width, aspect)
    height = width / aspect
    suffix = f"{width:.0f}x{height:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    num_iterations = 500

    input_path = Path(input_dir)
    output_path = Path(output_dir).joinpath(base_benchmark_name)
    output_path.mkdir(parents=True, exist_ok=True)

    METHOD_PRETTY_NAMES = {
        "random": "Random",
        "tpe": "TPE",
        "bore": "BORE",
    }

    frames = []
    for d in dimensions:

        benchmark_name = f"{base_benchmark_name}_{d:03d}d"
        error_min = get_error_mins(benchmark_name, input_dir)

        for method in methods:

            for run in range(num_runs):

                path = input_path.joinpath(benchmark_name, method, f"{run:03d}.csv")
                frame = load_frame(path, run, error_min=error_min)
                frames.append(frame.assign(method=method, d=d))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    data.replace(dict(method=METHOD_PRETTY_NAMES), inplace=True)
    data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)

    hue_order = style_order = list(map(METHOD_PRETTY_NAMES.get, methods))

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="d", y="regret best", hue="method", hue_order=hue_order,
                 style="method", style_order=style_order,
                 err_kws=dict(edgecolor='none'),
                 data=data.query(f"iteration == {num_iterations-1}"), ax=ax)

    ax.set_xlabel(r"$D$")
    ax.set_ylabel(f"final regret (after {num_iterations} evaluations)")

    # ax.set_yscale("log")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"line_regret_dimensions_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.boxplot(x="d", y="regret best", hue="method", hue_order=hue_order,
                data=data.query(f"iteration == {num_iterations-1}"), ax=ax)

    ax.set_xlabel(r"$D$")
    ax.set_ylabel(f"final regret (after {num_iterations} evaluations)")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"box_regret_dimensions_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
