import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                FCNetSliceLocalizationBenchmark,
                                FCNetNavalPropulsionBenchmark,
                                FCNetParkinsonsTelemonitoringBenchmark)


GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499
ERROR_MINS = dict(
    branin=0.397887,
    hartmann3d=-3.86278,
    hartmann6d=-3.32237,
    borehole=-309.5755876604079,
    styblinski_tang=-39.16599
)


def pt_to_in(x):
    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):
    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def sanitize(data, mapping):

    return data.replace(dict(method=mapping)) \
               .rename(lambda s: s.replace('_', ' '), axis="columns")


def get_ci(ci):

    if ci is not None:

        try:
            return int(ci)
        except ValueError:
            assert ci == "sd", "if `ci` is not integer and not None, " \
                "it must be 'sd'!"
    return ci


def get_error_mins(benchmark_name, input_dir, data_dir=None):

    base_path = Path(input_dir).joinpath(benchmark_name)

    if benchmark_name.startswith("fcnet"):

        path = base_path.joinpath("minimum.yaml")

        if path.exists():

            with path.open('r') as f:
                val_error_min = yaml.safe_load(f).get("val_error_min")

        else:
            assert data_dir is not None, "data directory must be specified"

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

            config_dict, \
                val_error_min, test_error_min = benchmark.get_best_configuration()

            d = dict(config_dict=config_dict,
                     val_error_min=float(val_error_min),
                     test_error_min=float(test_error_min))

            with path.open('w') as f:
                yaml.dump(d, f)

        return float(val_error_min)

    if benchmark_name.startswith("styblinski_tang"):
        *head, dimensions_str = benchmark_name.split('_')
        dimensions = int(dimensions_str[:-1])
        return dimensions * ERROR_MINS.get("styblinski_tang")

    if benchmark_name.startswith("michalewicz"):

        path = base_path.joinpath("L-BFGS-B", "minimum.yaml")
        with path.open('r') as f:
            error_min = yaml.safe_load(f).get('y')
        return error_min

    return ERROR_MINS.get(benchmark_name)


def load_frame(path, run, loss_min=None, loss_key="loss", sort_key="finished",
               duration_key=None):

    frame = pd.read_csv(path, index_col=0)

    # sort and drop old index
    frame.sort_values(by=sort_key, axis="index", ascending=True, inplace=True)
    frame.reset_index(drop=True, inplace=True)

    loss = frame[loss_key]
    best = loss.cummin()

    if duration_key is None:
        elapsed = frame["finished"]
    else:
        duration = frame[duration_key]
        elapsed = duration.cumsum()

    frame = frame.assign(run=run, evaluation=frame.index + 1, best=best,
                         elapsed=elapsed)

    if "epoch" in frame.columns:
        target = frame.groupby(by="task").epoch.max()
        resource = frame.epoch.cumsum()
        frame = frame.assign(target=target, resource=resource)

    if loss_min is not None:
        error = loss.sub(loss_min).abs()
        regret = error.cummin()
        frame = frame.assign(error=error, regret=regret)

    return frame


def extract_series(frame, index="elapsed", column="regret"):

    frame_new = frame.set_index(index)
    series = frame_new[column]

    # (0) save last timestamp and value
    # tail = series.tail(n=1)
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
    # .append(tail)
    return series_new


def merge_stack_series(series_dict, run_key="run", y_key="regret"):

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
