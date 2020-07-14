"""Datasets module."""
import pandas as pd
import json


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
