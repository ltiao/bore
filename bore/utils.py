import pandas as pd

from pathlib import Path


def preprocess(df):

    new_df = df.assign(timestamp=pd.to_datetime(df.timestamp, unit='s'))
    elapsed_delta = new_df.timestamp - new_df.timestamp.min()

    return new_df.assign(elapsed=elapsed_delta.dt.total_seconds())


def extract_series(df, index="elapsed", column="nelbo"):

    new_df = df.set_index(index)
    series = new_df[column]

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
    new_series = series.drop_duplicates(keep="first") \
                       .groupby(level=index).min()
#                        .append(series_final)

    return new_series


def merge_stack_runs(series_dict, seed_key="seed", y_key="nelbo",
                     drop_until_all_start=False):

    merged_df = pd.DataFrame(series_dict)

    # fill missing values by propagating previous observation
    merged_df.ffill(axis="index", inplace=True)

    # NaNs can only remain if there are no previous observations
    # i.e. these occur at the beginning rows.
    # drop rows until all runs have recorded observations.
    if drop_until_all_start:
        merged_df.dropna(how="any", axis="index", inplace=True)

    # TODO: Add option to impute with row-wise mean, which looks something like:
    #    (values in Pandas can only be filled column-by-column, so need to
    #     transpose, fillna and transpose back)
    # merged_df = merged_df.T.fillna(merged_df.mean(axis="columns")).T

    merged_df.columns.name = seed_key
    stacked_df = merged_df.stack(level=seed_key)

    stacked_df.name = y_key
    data = stacked_df.reset_index()

    return data


def make_plot_data(names, seeds, summary_dir,
                   process_run_fn=None,
                   extract_series_fn=None,
                   seed_key="seed",
                   y_key="nelbo"):

    base_path = Path(summary_dir)

    if process_run_fn is None:

        def process_run_fn(run_df):
            return run_df

    df_list = []

    for name in names:

        path = base_path.joinpath(name)
        seed_dfs = dict()

        for seed in seeds:

            csv_path = path.joinpath(f"scalars.{seed:03d}.csv")
            seed_df = pd.read_csv(csv_path)

            seed_dfs[seed] = process_run_fn(seed_df)

        if extract_series_fn is not None:

            series_dict = {seed: extract_series_fn(seed_df)
                           for seed, seed_df in seed_dfs.items()}

            name_df = merge_stack_runs(series_dict, seed_key=seed_key,
                                       y_key=y_key).assign(name=name)

        else:

            name_df = pd.concat(seed_dfs.values(), axis="index", sort=True)

        df_list.append(name_df)

    data = pd.concat(df_list, axis="index", sort=True)

    return data
