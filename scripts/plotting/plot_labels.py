import sys
import click

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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
@click.option('--num-iters', default=25)
@click.option('--quantile', default=1/3)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
@click.option('--seed', default=42)
def main(name, num_iters, quantile, context, style, palette, width, aspect,
         extension, output_dir, seed):

    figsize = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    random_state = np.random.RandomState(seed)

    Y = np.empty((num_iters, num_iters))
    Z = np.empty((num_iters, num_iters), dtype=bool)
    Q = np.empty((num_iters, num_iters))
    # %%

    ys = []
    for i in range(num_iters):

        y = random_state.randn()
        ys.append(y)

        a = np.asarray(ys)
        t = np.quantile(a, q=quantile)

        Y[i, :i+1] = a
        Z[i, :i+1] = np.less_equal(a, t)
        Q[i, :i+1] = np.argsort(np.argsort(a)) / i if i > 0 else 0.

    # %%
    mask = np.zeros_like(Y)
    mask[np.triu_indices_from(mask, k=1)] = True

    # %%
    fig, ax = plt.subplots()

    sns.heatmap(Y, mask=mask, annot=True, fmt=".02f", linewidths=1.0,
                cbar_kws=dict(orientation="horizontal"),
                cmap="cividis_r", square=True, ax=ax)

    ax.set_xlabel("index")
    ax.set_ylabel("iteration")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"numerical_{context}_{suffix}.{ext}"))

    plt.show()

    # %%
    fig, ax = plt.subplots()

    sns.heatmap(Q, mask=mask, annot=True, fmt=".02f", linewidths=1.0,
                cbar_kws=dict(orientation="horizontal"),
                cmap="cividis_r", square=True, ax=ax)

    ax.set_xlabel("index")
    ax.set_ylabel("iteration")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"quantiles_{context}_{suffix}.{ext}"))

    plt.show()

    # %%
    fig, ax = plt.subplots()

    sns.heatmap(Z, mask=mask, cbar_kws=dict(orientation="horizontal"),
                linewidths=1.0, cmap="cividis", square=True, ax=ax)

    ax.set_xlabel("index")
    ax.set_ylabel("iteration")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"binary_{context}_{suffix}.{ext}"))

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
