import sys
import click

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

from scipy.optimize import Bounds

from bore.benchmarks import Forrester
from bore.models import MaximizableSequential

from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import GOLDEN_RATIO, WIDTH, pt_to_in


def f(num_train, batch_size, num_steps):

    steps_per_epoch = np.ceil(np.true_divide(num_train, batch_size))
    num_epochs = num_steps // steps_per_epoch

    return num_epochs


@click.command()
@click.argument("name")
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
def main(name, output_dir, transparent, context, style,
         palette, width, height, aspect, dpi, extension):

    log_batch_sizes = np.arange(3, 7)
    batch_sizes = 2**log_batch_sizes
    iteration = np.arange(1, 2000)

    # preamble
    if height is None:
        height = width / aspect
    # height *= num_iterations
    # figsize = size(width, aspect)
    figsize = (width, height)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": False,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)
    # / preamble

    frames = []

    for batch_size in batch_sizes:
        for i in range(1, 5):
            num_steps = 200 * i
            epochs = f(iteration, batch_size=batch_size, num_steps=num_steps)
            frame = pd.DataFrame(dict(batch_size=batch_size,
                                      iteration=iteration,
                                      num_steps=num_steps,
                                      epochs=epochs))
            frames.append(frame)

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
    data.rename(columns=dict(batch_size="batch size", num_steps="steps per iteration"), inplace=True)

    g = sns.relplot(x="iteration", y="epochs", hue="steps per iteration",
                    col="batch size", kind="line", palette=palette,
                    height=height, aspect=aspect, data=data)
    g = g.set(yscale="log")
    for ext in extension:
        g.savefig(output_path.joinpath(f"decay_{context}_{suffix}.{ext}"))

    # fig, ax = plt.subplots()

    # sns.lineplot(x="iteration", y="epochs", hue="batch_size",
    #              palette="deep", data=data, ax=ax)

    # plt.tight_layout()

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"decay_{context}_{suffix}.{ext}"),
    #                 dpi=dpi, transparent=transparent)

    # plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
