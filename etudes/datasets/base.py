"""Datasets module."""

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

import json

from pathlib import Path
from .decorators import binarize

SEED = 42

tfd = tfp.distributions


def read_fcnet_data(f, max_configs=None, num_seeds=4):

    frames = []

    for config_num, config_str in enumerate(f.keys()):

        if max_configs is not None and config_num > max_configs:
            break

        config = json.loads(config_str)

        for seed in range(num_seeds):

            config["seed"] = seed

            for attr in f[config_str].keys():
                config[attr] = f[config_str][attr][seed]

            frame = pd.DataFrame(config)
            frame.index.name = "epoch"
            frame.reset_index(inplace=True)

            frames.append(frame)

    return pd.concat(frames, axis="index", ignore_index=True, sort=True)


@binarize(positive_label=2, negative_label=7)
def binary_mnist_load_data():
    return tf.keras.datasets.mnist.load_data()


def get_sequence_path(sequence_num, base_dir="../datasets"):

    return Path(base_dir).joinpath("bee-dance", "zips", "data",
                                   f"sequence{sequence_num:d}", "btf")


bee_dance_filenames = dict(
    x="ximage.btf",
    y="yimage.btf",
    t="timage.btf",
    label="label0.btf",
    timestamp="timestamp.btf"
)


def read_sequence_column(sequence_num, col_name, base_dir="../datasets"):

    sequence_path = get_sequence_path(sequence_num, base_dir=base_dir)

    return pd.read_csv(sequence_path / bee_dance_filenames[col_name],
                       names=[col_name], header=None)


def read_sequence(sequence_num, base_dir="../datasets"):

    left = None

    for col_name in bee_dance_filenames:

        right = read_sequence_column(sequence_num, col_name, base_dir=base_dir)

        if left is None:
            left = right
        else:
            left = pd.merge(left, right, left_index=True, right_index=True)

    change_point = left.label != left.label.shift()
    phase = change_point.cumsum()

    return left.assign(change_point=change_point, phase=phase)


def load_bee_dance_dataframe(base_dir="../datasets"):

    sequences = []

    for i in range(6):

        sequence_num = i + 1
        sequence = read_sequence(sequence_num, base_dir=base_dir) \
            .assign(sequence=sequence_num)
        sequences.append(sequence)

    return pd.concat(sequences, axis="index")


def coal_mining_disasters_load_data(base_dir="../datasets/"):
    """
    Coal mining disasters dataset.

    Examples
    --------

    .. plot::
        :context: close-figs

        from etudes.datasets import coal_mining_disasters_load_data

        X, y = coal_mining_disasters_load_data()

        fig, ax = plt.subplots()

        ax.vlines(X.squeeze(), ymin=0, ymax=y, linewidth=0.5, alpha=0.8)

        ax.set_xlabel("days")
        ax.set_ylabel("incidents")

        plt.show()
    """
    base = Path(base_dir).joinpath("coal-mining-disasters")

    data = pd.read_csv(base / "data.csv", names=["count", "days"], header=None)

    X = np.expand_dims(data["days"].values, axis=-1)
    y = data["count"].values

    return X, y


def mauna_loa_load_dataframe(base_dir="../datasets/"):
    """
    Mauna Loa dataset.

    Examples
    --------

    .. plot::
        :context: close-figs

        import seaborn as sns
        from etudes.datasets import mauna_loa_load_dataframe

        data = mauna_loa_load_dataframe()

        g = sns.relplot(x='date', y='average', kind="line",
                        data=data, height=5, aspect=1.5, alpha=0.8)
        g.set_ylabels(r"average $\mathrm{CO}_2$ (ppm)")
    """
    base = Path(base_dir).joinpath("mauna-loa-co2")

    column_names = ["year", "month", "date", "average", "interpolated",
                    "trend", "num_days"]

    data = pd.read_csv(base / "co2_mm_mlo.txt", names=column_names,
                       comment="#", header=None, sep=r"\s+")
    data = data[data.average > 0]

    return data
