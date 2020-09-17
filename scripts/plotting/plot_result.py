import sys
import click

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from utils import (GOLDEN_RATIO, WIDTH, size, load_frame, extract_series,
                   merge_stack_series, get_error_mins)

OUTPUT_DIR = "figures/"


@click.command()
@click.argument("benchmark_name")
@click.argument("input_dir", default="results",
                type=click.Path(file_okay=False, dir_okay=True))
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
def main(benchmark_name, input_dir, num_runs, methods, ci, context, style,
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

    input_path = Path(input_dir)
    output_path = Path(output_dir).joinpath(benchmark_name)
    output_path.mkdir(parents=True, exist_ok=True)

    METHOD_PRETTY_NAMES = {
        "random": "Random",
        "tpe": "TPE",
        "bore": "BORE",
        # "boredom": "BORE II",
        # "boredom-real": "BORE III",
        "bore-sigmoid-elu-ftol-1e-2-gamma-0.33333333333333333333": "BORE",
        "boredom-real": "BORE",
        "bore-logit-elu-ftol-1e-2-gamma-0.33333333333333333333": "BORE",
        "bore-sigmoid-elu-ftol-1e-9-random-0.1": "BORE"
    }

    error_min = get_error_mins(benchmark_name, input_dir, data_dir="datasets/fcnet_tabular_benchmarks")

    frames = []
    for method in methods:

        # series = {}
        for run in range(num_runs):

            path = input_path.joinpath(benchmark_name, method, f"{run:03d}.csv")
            frame = load_frame(path, run, error_min=error_min)
            frames.append(frame.assign(method=method))
            # series[run] = extract_series(frame, index="elapsed", column="regret_best")

        # frame = merge_stack_series(series).assign(method=method)
        # frames.append(frame)

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    data.replace(dict(method=METHOD_PRETTY_NAMES), inplace=True)
    data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)

    print(data)

    hue_order = style_order = list(map(METHOD_PRETTY_NAMES.get, methods))

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="iteration", y="regret best",
                 hue="method", hue_order=hue_order,
                 style="method", style_order=style_order,
                 # units="run", estimator=None,
                 # ci=None,
                 err_kws=dict(edgecolor='none'),
                 data=data, ax=ax)

    ax.set_xlabel("evaluations")
    ax.set_ylabel("simple regret")

    ax.set_yscale("log")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"regret_iterations_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # fig, ax = plt.subplots()
    # sns.despine(fig=fig, ax=ax, top=True)

    # sns.lineplot(x="elapsed", y="regret best",
    #              hue="method", hue_order=hue_order,
    #              style="method", style_order=style_order,
    #              # units="run", estimator=None,
    #              # ci=None,
    #              err_kws=dict(edgecolor='none'),
    #              data=data, ax=ax)

    # ax.set_xlabel("wall-clock time elapsed (s)")
    # ax.set_ylabel("incumbent regret")

    # # ax.set_xscale("log")
    # ax.set_yscale("log")

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"regret_elapsed_{context}_{suffix}.{ext}"),
    #                 bbox_inches="tight")

    # plt.show()

    # # g = sns.relplot(x="elapsed", y="regret", hue="run",
    # #                 col="method", palette="tab20",
    # #                 alpha=0.6, kind="scatter", data=data)
    # # g.map(sns.lineplot, "task", "regret best", "run",
    # #       palette="tab20", linewidth=2.0, alpha=0.8)
    # # g.set_axis_labels("iteration", "regret")

    # # for ext in extension:
    # #     g.savefig(output_path.joinpath(f"regret_vs_elapsed_all_{context}_{suffix}.{ext}"))

    # # g = sns.relplot(x="task", y="error", hue="epoch",
    # #                 col="run", col_wrap=4, palette="Dark2",
    # #                 alpha=0.6, kind="scatter", data=data.query("method == 'BORE'"))
    # # g.map(plt.plot, "task", "best", color="k", linewidth=2.0, alpha=0.8)
    # # g.set_axis_labels("iteration", "regret")

    # # for ext in extension:
    # #     g.savefig(output_path.joinpath(f"error_vs_iterations_{context}_{suffix}.{ext}"))

    # return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
