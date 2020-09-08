import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from utils import GOLDEN_RATIO, WIDTH, size


@click.command()
@click.argument("name")
@click.option("--seed", '-s', default=8888)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--input-dir", default="results",
              type=click.Path(file_okay=False, dir_okay=True))
@click.option("--output-dir", default="figures/",
              type=click.Path(file_okay=False, dir_okay=True))
def main(name, seed, context, style, palette, width, aspect, extension,
         input_dir, output_dir):

    figsize = width_in, height_in = size(width, aspect)
    height = width / aspect
    suffix = f"{width:.0f}x{height:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir).joinpath(name)
    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    # t_grid = np.arange(num_epochs)

    # # lr_grid = np.logspace(-5, -0.5, num_index_points)
    # log_lr_grid = np.linspace(-5.0, -1.0, num_index_points)
    # lr_grid = 10**log_lr_grid

    data = pd.read_csv(input_path.joinpath(f"{seed:04d}.csv"), index_col=0)
    data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)
    data = data.assign(epoch=lambda n: n+1)

    data_pivot = data.pivot(index="log lr", columns="epoch", values="val nmse")
    Z = data_pivot.to_numpy()

    fig, ax = plt.subplots()

    sns.lineplot(x="epoch", y="val nmse", hue="log lr",
                 units="seed", estimator=None,
                 palette="viridis_r", linewidth=0.2, data=data, ax=ax)

    ax.set_xscale("log", base=3)
    ax.set_yscale("log")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"metric_epoch_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    fig, ax = plt.subplots()

    sns.lineplot(x="lr", y="val nmse", hue="epoch",
                 units="seed", estimator=None,
                 palette="viridis_r", linewidth=0.2, data=data, ax=ax)

    ax.set_xscale("log")
    ax.set_yscale("log")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"metric_lr_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    epoch_grid = data_pivot.columns.to_numpy()
    log_lr_grid = data_pivot.index.to_numpy()

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=50))

    ax.plot_surface(log_lr_grid.reshape(-1, 1), epoch_grid, np.log10(Z),
                    alpha=0.8, edgecolor='k', linewidth=0.4, cmap="Spectral_r")

    ax.set_xlabel(r"$\log_{10}$ learning rate")
    ax.set_ylabel("epoch")
    ax.set_zlabel(r"$\log_{10}$ val nmse")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"learning_curves_{context}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    b_max = data.epoch.max()
    T = int(np.ceil(np.log(b_max) / np.log(3)))
    t_grid = np.arange(T+1)
    columns = list(np.minimum(3**t_grid, b_max))

    test = data_pivot[columns]  # .applymap(np.log10)
    test.columns = [fr"$f(x, {{{t+1:d}}})$" for t in t_grid]

    g = sns.PairGrid(test, corner=True, height=height_in/(T+1), aspect=aspect)
    g = g.map_lower(plt.plot, alpha=0.7)

    for ext in extension:
        g.savefig(output_path.joinpath(f"grid_line_{context}_{suffix}.{ext}"))

    g = sns.PairGrid(test.reset_index(), hue="log lr", palette="viridis",
                     corner=True, height=height_in/(T+1), aspect=aspect)
    g = g.map_lower(plt.scatter, facecolor="none", alpha=0.8)

    for ext in extension:
        g.savefig(output_path.joinpath(f"grid_scatter_{context}_{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
