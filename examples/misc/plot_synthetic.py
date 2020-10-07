# -*- coding: utf-8 -*-
"""
Multi-start Newton hill-climbing optimization
=============================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from bore.benchmarks import GoldsteinPrice, Ackley
# %%

num_runs = 50
num_iterations = 1000

seed = 8888  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

benchmark = Ackley(dimensions=2)

cs = benchmark.get_config_space()
hx = cs.get_hyperparameter("x0")
hy = cs.get_hyperparameter("x1")

# constants
x_min, x_max = hx.lower, hx.upper
y_min, y_max = hy.lower, hy.upper

y, x = np.ogrid[y_min:y_max:200j, x_min:x_max:200j]
X, Y = np.broadcast_arrays(x, y)

# def func(x, y):
#     return benchmark(np.dstack([x, y]))
# %%


fig, ax = plt.subplots()

contours = ax.contour(X, Y, benchmark.func(X, Y), levels=np.logspace(0, 6, 20),
                      norm=LogNorm(), cmap="turbo")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt='%.1f')


ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

plt.show()
# %%
fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=-135, elev=35))

ax.plot_surface(x, y, benchmark.func(X, Y), edgecolor='k', linewidth=0.5, cmap="turbo")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")

plt.show()
# %%

# # constants
# x_min, x_max = -5., 5.

# y, x = np.ogrid[x_min:x_max:200j, x_min:x_max:200j]
# X, Y = np.broadcast_arrays(x, y)


# def func(x, y):
#     return styblinski_tang(np.dstack([x, y]))
# # %%


# fig, ax = plt.subplots()

# ax.contour(X, Y, func(X, Y), cmap="Spectral_r")

# ax.set_xlabel(r"$x_1$")
# ax.set_ylabel(r"$x_2$")

# plt.show()
# # %%
# fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=-135, elev=35))

# ax.plot_surface(x, y, func(X, Y), edgecolor='k', linewidth=0.5, cmap="Spectral_r")

# ax.set_xlabel(r"$x_1$")
# ax.set_ylabel(r"$x_2$")
# ax.set_zlabel(r"$y$")

# plt.show()
# # %%

# frames = []
# for dim in range(2, 20, 2):

#     xs = x_min + (x_max - x_min) * random_state.rand(num_runs, num_iterations, dim)
#     ys = styblinski_tang(xs)
#     y_min = -39.16599 * dim

#     df = pd.DataFrame(np.abs(y_min - np.minimum.accumulate(ys, axis=1)))
#     df.index.name = "run"
#     df.columns.name = "iteration"

#     s = df.stack()
#     s.name = "regret"

#     frame = s.reset_index()
#     frames.append(frame.assign(name=rf"$D={dim}$"))
# # %%
# data = pd.concat(frames, axis="index", sort=True)
# # %%

# fig, ax = plt.subplots()

# sns.lineplot(x="iteration", y="regret", hue="name",
#              # units="run", estimator=None,
#              ci="sd", palette="deep", data=data, ax=ax)
# ax.set_yscale("log")

# plt.show()
# # %%

# fig, ax = plt.subplots()

# sns.boxplot(x="name", y="regret", palette="deep",
#             data=data.query(f"iteration == {num_iterations-1}"), ax=ax)

# ax.set_ylabel(f"final regret (after {num_iterations} evaluations)")

# plt.show()
# # %%

# fig, ax = plt.subplots()

# sns.lineplot(x="name", y="regret", hue="iteration", palette="viridis_r",
#              ci=None, linewidth=0.1, data=data, ax=ax)

# plt.show()
# # %%

# fig, ax = plt.subplots()

# sns.lineplot(x="name", y="regret", ci='sd',
#              data=data.query(f"iteration == {num_iterations-1}"), ax=ax)

# ax.set_ylabel(f"final regret (after {num_iterations} evaluations)")

# plt.show()
