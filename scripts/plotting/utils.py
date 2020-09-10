import numpy as np
import pandas as pd

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499


def pt_to_in(x):
    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):
    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def load_frame(path, run, error_min=None, sort_by="finished"):

    frame = pd.read_csv(path, index_col=0)

    # sort and drop old index
    frame.sort_values(by=sort_by, axis="index", ascending=True, inplace=True)
    frame.reset_index(drop=True, inplace=True)

    error = frame["loss"]
    duration = frame["info"]

    best = error.cummin()
    elapsed = duration.cumsum()

    target = frame.groupby(by="task").epoch.max()
    resource = frame.epoch.cumsum()

    frame = frame.assign(run=run, iteration=frame.index + 1,
                         best=best, elapsed=elapsed, target=target,
                         resource=resource)

    if error_min is not None:
        regret = error.sub(error_min).abs()
        regret_best = regret.cummin()
        frame = frame.assign(regret=regret, regret_best=regret_best)

    return frame


def extract_series(frame, index="elapsed", column="regret_best"):

    frame_new = frame.set_index(index)
    series = frame_new[column]

    # (0) save last timestamp and value
    tail = series.tail(n=1)
    # (1) de-duplicate the values (significantly speed-up
    # subsequent processing)
    # (2) de-duplicate the indices (it is entirely possible
    # for some epoch of two different tasks to complete
    # at the *exact* same time; we take the one with the
    # smaller value)
    # (3) add back last timestamp and value which can get
    # lost in step (1)
    series_new = series.drop_duplicates(keep="first") \
                       .groupby(level=index).min() \
                       .append(tail)
    return series_new


def merge_stack_series(series_dict, run_key="run", y_key="regret_best"):

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
