import sys
import click

import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.metrics import mean_squared_error

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from pathlib import Path
from tqdm import tqdm


def nmse(y_test, y_pred):

    return mean_squared_error(y_test, y_pred) / K.var(y_test)


@click.command()
@click.argument("name")
@click.option("--num-index-points", '-n', default=64)
@click.option("--num-layers", default=3)
@click.option("--num-units", default=32)
@click.option("--num-epochs", default=243)
@click.option("--batch-size", default=64)
@click.option("--val-rate", default=0.2)
@click.option("--seed", '-s', default=8888)
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, num_index_points, num_layers, num_units, num_epochs, batch_size,
         val_rate, seed, output_dir):

    random_state = np.random.RandomState(seed)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    # lr_grid = np.logspace(-5, -0.5, num_index_points)
    log_lr_grid = np.linspace(-5.0, -1.0, num_index_points)
    lr_grid = 10**log_lr_grid

    dataset = load_boston()

    X_train, X_val, y_train, y_val = train_test_split(dataset.data,
                                                      dataset.target,
                                                      test_size=val_rate,
                                                      random_state=random_state)

    frames = []
    for i, lr in enumerate(tqdm(lr_grid)):

        model = Sequential()
        for _ in range(num_layers):
            model.add(Dense(num_units, activation="relu",
                            kernel_initializer=GlorotUniform(seed=seed)))
        model.add(Dense(1, kernel_initializer=GlorotUniform(seed=seed)))

        optimizer = Adam(learning_rate=lr)

        model.compile(optimizer=optimizer, loss="mean_squared_error",
                      metrics=[nmse])
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=num_epochs, batch_size=batch_size, verbose=False)

        frame = pd.DataFrame(hist.history).assign(log_lr=log_lr_grid[i],
                                                  lr=lr, seed=seed)
        frame.index.name = "epoch"
        frame.reset_index(inplace=True)
        frames.append(frame)

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
    data.to_csv(output_path.joinpath(f"{seed:04d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
