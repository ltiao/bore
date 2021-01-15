import sys
import click

import requests
import json

import matplotlib.pyplot as plot
import seaborn as sns

from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import GOLDEN_RATIO, WIDTH, pt_to_in


@click.command()
@click.argument("name")
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option("--symbol", "-s", default="ETHAUD")
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option('--seed', '-s', default=1)
def main(name, output_dir, symbol, transparent, context, style,
         palette, width, height, aspect, dpi, extension, seed):

    next_loc_color = "tab:green"

    num_index_points = 512
    num_random_init = 4
    num_features = 1

    gamma = 0.25

    num_units = 32
    activation = "softplus"
    optimizer = "adam"
    epochs = 500
    batch_size = 64

    noise_scale = 0.05

    random_state = np.random.RandomState(seed)

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

    r = requests.get("https://api.binance.com/api/v3/depth", params=dict(symbol=symbol))

    fig, ax = plt.subplots()
    click.echo(r.json())

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"frame_{i:02d}_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
