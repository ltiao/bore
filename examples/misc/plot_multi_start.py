# -*- coding: utf-8 -*-
"""
Multi-start Newton hill-climbing optimization
=============================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 3
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import Bounds

from bore.optimizers import multi_start_lbfgs_minimizer
from bore.decorators import value_and_gradient, numpy_io
# %%

# constants
# num_index_points = 128
y_min, y_max = 0, 15
x_min, x_max = -5, 10

y, x = np.ogrid[y_min:y_max:200j, x_min:x_max:200j]
X, Y = np.broadcast_arrays(x, y)

dim = 2

seed = 8888  # set random seed for reproducibility
random_state = np.random.RandomState(seed)
# %%


def unpack(fn):

    def new_fn(xs):

        return fn(*xs)

    return new_fn
# %%


def branin(x, y, a=1.0, b=5.1/(4*np.pi**2), c=5.0/np.pi, r=6.0, s=10.0,
           t=1.0/(8*np.pi)):
    return a*(y - b * x**2 + c*x - r)**2 + s*(1 - t)*tf.cos(x) + s
# %%


# def currin(x1, x2):

#     a = 2300*x1**3 + 1900*x1**2 + 2092*x1 + 60
#     b = 100*x1**3 + 500*x1**2 + 4*x1 + 20
#     c = 1 - tf.exp(-0.5/x2)
#     return c * a / b
# %%
fig, ax = plt.subplots()

ax.contour(X, Y, branin(x, y), levels=np.logspace(0, 5, 35), norm=LogNorm(),
           cmap="Spectral_r")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

plt.show()
# %%
fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=-135, elev=35))

ax.plot_surface(x, y, branin(x, y), edgecolor='k', linewidth=0.5, cmap="Spectral_r")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")

plt.show()
# %%


@numpy_io
@value_and_gradient
@unpack
def func(x1, x2):

    return branin(x1, x2)
# %%


bounds = Bounds(lb=[x_min, y_min], ub=[x_max, y_max])
results = multi_start_lbfgs_minimizer(func, bounds, random_state=random_state)

len(results)
# %%

U = np.vstack([res.x for res in results])
v = np.hstack([res.fun for res in results])
# %%

fig, ax = plt.subplots()

ax.scatter(*U.T, c=v, cmap="Spectral_r")
ax.contour(X, Y, branin(x, y), levels=np.logspace(0, 5, 35), norm=LogNorm(),
           cmap="Spectral_r")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

plt.show()
# %%

res_best = min(filter(lambda res: res.success, results), key=lambda res: res.fun)
res_best
