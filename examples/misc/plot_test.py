# -*- coding: utf-8 -*-
"""
Test
====

Hello world
"""
# sphinx_gallery_thumbnail_number = 3

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from bore.utils import load_runs
from pathlib import Path
# %%

# constants
num_runs = 20
error_min = -3.32237
optimizers = ["random", "tpe", "bore", "gamma15"]
base_dir = Path("../../results")
# %%
frames = []

for optimizer in optimizers:

    frame = load_runs(base_dir.joinpath(optimizer), runs=num_runs, error_min=error_min)
    frames.append(frame.assign(optimizer=optimizer))
# %%
data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)
# %%
# Random Search
# -------------
g = sns.relplot(x="task", y="regret", hue="epoch",
                col="run", col_wrap=4, palette="Dark2",
                alpha=0.6, kind="scatter", data=data.query("optimizer == 'random'"))
g.map(plt.plot, "task", "regret best", color="k", linewidth=2.0, alpha=0.8)
g.set_axis_labels("iteration", "regret")
# g.set(xscale="log", yscale="log")
# %%
# Tree Parzen Estimator (TPE)
# ---------------------------
g = sns.relplot(x="task", y="regret", hue="epoch",
                col="run", col_wrap=4, palette="Dark2",
                alpha=0.6, kind="scatter", data=data.query("optimizer == 'tpe'"))
g.map(plt.plot, "task", "regret best", color="k", linewidth=2.0, alpha=0.8)
g.set_axis_labels("iteration", "regret")
# g.set(xscale="log", yscale="log")
# %%
# BO via Density Ratio Estimation (BORE)
# --------------------------------------
g = sns.relplot(x="task", y="regret", hue="epoch",
                col="run", col_wrap=4, palette="Dark2",
                alpha=0.6, kind="scatter", data=data.query("optimizer == 'bore'"))
g.map(plt.plot, "task", "regret best", color="k", linewidth=2.0, alpha=0.8)
g.set_axis_labels("iteration", "regret")
# g.set(xscale="log", yscale="log")
# %%
g = sns.relplot(x="task", y="regret", hue="run",
                col="optimizer", palette="tab20",
                alpha=0.6, kind="scatter", data=data)
g.map(sns.lineplot, "task", "regret best", "run",
      palette="tab20", linewidth=2.0, alpha=0.8)
g.set_axis_labels("iteration", "regret")
# g.set(xscale="log", yscale="log")
# %%
fig, ax = plt.subplots()

sns.lineplot(x="task", y="regret best", hue="optimizer",
             ci="sd", data=data, ax=ax)

ax.set_xlabel("iteration")
ax.set_ylabel("incumbent regret")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-1, -error_min)

plt.show()
