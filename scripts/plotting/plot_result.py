import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# from bore.utils import load_runs
from pathlib import Path

import ConfigSpace as CS
from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                FCNetSliceLocalizationBenchmark,
                                FCNetNavalPropulsionBenchmark,
                                FCNetParkinsonsTelemonitoringBenchmark)

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def extract_series(frame, index="elapsed", column="regret_best"):

    frame_new = frame.set_index(index)
    series = frame_new[column]

    # (0) save last timestamp and value
#     series_final = series.tail(n=1)

    # (1) de-duplicate the values (significantly speed-up
    # subsequent processing)
    # (2) de-duplicate the indices (it is entirely possible
    # for some epoch of two different tasks to complete
    # at the *exact* same time; we take the one with the
    # smaller value)
    # (3) add back last timestamp and value which can get
    # lost in step (1)
    series_new = series.drop_duplicates(keep="first") \
                       .groupby(level=index).min()
#                        .append(series_final)
    return series_new


def load_runs(base_dir, runs=[], error_min=None):

    base_path = Path(base_dir)

    series = {}
    for run in range(runs):

        path = base_path.joinpath(f"{run:03d}.csv")
        frame = pd.read_csv(path, index_col=0).assign(run=run)

        best = frame.error.cummin()
        elapsed = frame.cost.cumsum()
        frame = frame.assign(best=best, elapsed=elapsed)

        if error_min is not None:
            regret = (error_min - frame.error).abs()
            regret_best = regret.cummin()
            frame = frame.assign(regret=regret, regret_best=regret_best)

        series[run] = extract_series(frame, index="elapsed", column="regret_best")

    return series  # pd.concat(frames, axis="index", ignore_index=True, sort=True)


def merge_stack_runs(series_dict, run_key="run", y_key="regret_best"):

    frame = pd.DataFrame(series_dict)

    # fill missing values by propagating previous observation
    frame.ffill(axis="index", inplace=True)

    # NaNs can only remain if there are no previous observations
    # i.e. these occur at the beginning rows.
    # drop rows until all runs have recorded observations.
    frame.dropna(how="any", axis="index", inplace=True)

    frame.columns.name = run_key
    stacked = frame.stack(level=run_key)

    stacked.name = y_key
    stacked_frame = stacked.reset_index()

    return stacked_frame


def get_error_mins(benchmark_name, data_dir=None, budget=100):

    print(benchmark_name)

    if not benchmark_name.startswith("fcnet"):

        errors_mins = {
            "branin": 0.397887,
            "hartmann3d": -3.86278,
            "hartmann6d": -3.32237,
            "borehole": -309.5755876604079,
        }
        return errors_mins.get(benchmark_name)

    assert data_dir is not None, "data directory must be specified"

    x_mins = {
        "fcnet_protein": {
            "init_lr": 5 * 1e-4,
            "batch_size": 8,
            "lr_schedule": "cosine",
            "activation_fn_1": "relu",
            "activation_fn_2": "relu",
            "n_units_1": 512,
            "n_units_2": 512,
            "dropout_1": 0.0,
            "dropout_2": 0.3
        },
        "fcnet_slice": {
            "init_lr": 5 * 1e-4,
            "batch_size": 32,
            "lr_schedule": "cosine",
            "activation_fn_1": "relu",
            "activation_fn_2": "tanh",
            "n_units_1": 512,
            "n_units_2": 512,
            "dropout_1": 0.0,
            "dropout_2": 0.0
        },
        "fcnet_naval": {
            "init_lr": 5 * 1e-4,
            "batch_size": 8,
            "lr_schedule": "cosine",
            "activation_fn_1": "tanh",
            "activation_fn_2": "relu",
            "n_units_1": 128,
            "n_units_2": 512,
            "dropout_1": 0.0,
            "dropout_2": 0.0
        },
        "fcnet_parkinsons": {
            "init_lr": 5 * 1e-4,
            "batch_size": 8,
            "lr_schedule": "cosine",
            "activation_fn_1": "tanh",
            "activation_fn_2": "relu",
            "n_units_1": 128,
            "n_units_2": 512,
            "dropout_1": 0.0,
            "dropout_2": 0.0
        }
    }

    if benchmark_name.endswith("protein"):
        benchmark = FCNetProteinStructureBenchmark(data_dir=data_dir)
    elif benchmark_name.endswith("slice"):
        benchmark = FCNetSliceLocalizationBenchmark(data_dir=data_dir)
    elif benchmark_name.endswith("naval"):
        benchmark = FCNetNavalPropulsionBenchmark(data_dir=data_dir)
    elif benchmark_name.endswith("parkinsons"):
        benchmark = FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir)
    else:
        raise ValueError("dataset name not recognized!")

    cs = benchmark.get_configuration_space()
    c = CS.Configuration(cs, values=x_mins[benchmark_name])

    y, runtime = benchmark.objective_function(c, budget=budget)
    # y_test, runtime_test = benchmark.objective_function_test(c)

    return y


WIDTH = 397.48499
OUTPUT_DIR = "figures/"


@click.command()
@click.argument("benchmark_name")
@click.argument("input_dir", default="results",
                type=click.Path(file_okay=False, dir_okay=True))
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
def main(benchmark_name, input_dir, methods, ci, context, style, palette,
         width, aspect, extension, output_dir):

    figsize = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

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
        "random": "Random Search",
        "tpe": "TPE",
        "bore": "BORE",
        "bore-elu-normalize-ftol": "BORE (elu)",
        "bore-relu-normalize-ftol": "BORE (relu)",
        "bore-relu-normalize-high-ftol": "BORE (relu high ftol)"
    }

    num_runs = 20
    # runs = list(range(20))
    # runs.pop(8)
    # runs.pop(13-1)

    error_min = get_error_mins(benchmark_name, data_dir="datasets/fcnet_tabular_benchmarks")

    frames = []
    for method in methods:

        for run in range(num_runs):

            path = input_path.joinpath(benchmark_name, method, f"{run:03d}.csv")
            frame = pd.read_csv(path, index_col=0).assign(run=run, method=method)

            try:
                loss = frame["loss"]
                cost = frame["info"]
            except KeyError:
                loss = frame["error"]
                cost = frame["cost"]

            best = loss.cummin()
            elapsed = cost.cumsum()
            frame = frame.assign(best=best, elapsed=elapsed)

            if error_min is not None:
                regret = (error_min - loss).abs()
                regret_best = regret.cummin()
                frame = frame.assign(regret=regret, regret_best=regret_best)

            frames.append(frame)

        # series = load_runs(input_path.joinpath(benchmark_name, method),
        #                    runs=num_runs, error_min=error_min)
        # frame = merge_stack_runs(series)
        # frames.append(frame.assign(method=method))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    print(data)

    data.replace(dict(method=METHOD_PRETTY_NAMES), inplace=True)
    # data.replace({"optimizer": {"protein_structure": "Protein Structure"}}, inplace=True)
    data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)

    hue_order = style_order = list(map(METHOD_PRETTY_NAMES.get, methods))

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="task", y="regret best",
                 hue="method", hue_order=hue_order,
                 style="method", style_order=style_order,
                 # units="run", estimator=None,
                 # ci=None,
                 err_kws=dict(edgecolor='none'),
                 data=data, ax=ax)

    # ax.set_xlabel("wall-clock time elapsed (s)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("incumbent regret")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_ylim(1e-1, -error_min)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"iteration_vs_elapsed_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # g = sns.relplot(x="elapsed", y="regret", hue="run",
    #                 col="method", palette="tab20",
    #                 alpha=0.6, kind="scatter", data=data)
    # g.map(sns.lineplot, "task", "regret best", "run",
    #       palette="tab20", linewidth=2.0, alpha=0.8)
    # g.set_axis_labels("iteration", "regret")

    # for ext in extension:
    #     g.savefig(output_path.joinpath(f"regret_vs_elapsed_all_{context}_{suffix}.{ext}"))

    # g = sns.relplot(x="task", y="error", hue="epoch",
    #                 col="run", col_wrap=4, palette="Dark2",
    #                 alpha=0.6, kind="scatter", data=data.query("method == 'BORE'"))
    # g.map(plt.plot, "task", "best", color="k", linewidth=2.0, alpha=0.8)
    # g.set_axis_labels("iteration", "regret")

    # for ext in extension:
    #     g.savefig(output_path.joinpath(f"error_vs_iterations_{context}_{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
