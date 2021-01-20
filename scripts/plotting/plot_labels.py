import sys
import click

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from utils import GOLDEN_RATIO, WIDTH, pt_to_in


@click.command()
@click.argument("name")
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option("--num-iterations", "-i", default=6)
@click.option('--quantile', "-q", default=1/3)
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option('--seed', '-s', default=42)
def main(name, output_dir, num_iterations, quantile, transparent, context,
         style, palette, width, height, aspect, dpi, extension, seed):

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
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)
    # / preamble

    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

    random_state = np.random.RandomState(seed)

    Y = np.empty((num_iterations, num_iterations))
    Z = np.empty((num_iterations, num_iterations), dtype=bool)
    Q = np.empty((num_iterations, num_iterations))
    # %%

    ys = []
    for i in range(num_iterations):

        eps = random_state.randn()
        y = eps
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
    fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)

    sns.heatmap(Y, mask=mask, annot=True, fmt=".02f", linewidths=1.0,
                cmap="winter_r", ax=ax, cbar_ax=cbar_ax,
                cbar_kws={"orientation": "horizontal"})

    ax.set_xlabel(r"$n$")
    ax.set_ylabel("iteration")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"numerical_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    # %%
    fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)

    sns.heatmap(Q, mask=mask, annot=True, fmt=".02f", linewidths=1.0,
                cmap="winter_r", ax=ax, cbar_ax=cbar_ax,
                cbar_kws={"orientation": "horizontal"})

    ax.set_xlabel(r"$n$")
    ax.set_ylabel("iteration")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"quantiles_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    # %%
    fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)

    sns.heatmap(Z, mask=mask, linewidths=1.0, cmap="winter",
                ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})

    ax.set_xlabel(r"$n$")
    ax.set_ylabel("iteration")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"labels_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
