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

from bore.optimizers import minimize_multi_start
from bore.decorators import value_and_gradient, numpy_io, unstack
# %%

# constants
# num_index_points = 128
y_min, y_max = 0, 15
x_min, x_max = -5, 10

y, x = np.ogrid[y_min:y_max:200j, x_min:x_max:200j]
X, Y = np.broadcast_arrays(x, y)

num_starts = 20
num_samples = 1024

seed = 8888  # set random seed for reproducibility
random_state = np.random.RandomState(seed)
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
@unstack
def func(x1, x2):

    return branin(x1, x2)
# %%
print(func(np.zeros(shape=(50, 2))))
# %%
print(func(np.zeros(2)))
# %%

bounds = Bounds(lb=[x_min, y_min], ub=[x_max, y_max])
results = minimize_multi_start(func, bounds,
                               num_starts=num_starts,
                               num_samples=num_samples,
                               method="L-BFGS-B", jac=True,
                               options=dict(maxiter=100, ftol=1e-2),
                               random_state=random_state)
# %%
len(results)
# %%

U = np.vstack([res.x for res in results])
v = np.hstack([res.fun for res in results])
# %%

fig, ax = plt.subplots()

ax.scatter(*U.T, c=v, alpha=0.6, cmap="crest")
ax.contour(X, Y, branin(x, y), levels=np.logspace(0, 5, 35), norm=LogNorm(),
           cmap="crest")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

plt.show()
# %%

res_best = min(filter(lambda res: res.success, results), key=lambda res: res.fun)
res_best
# %%
# Borehole


@numpy_io
@value_and_gradient
@unstack
def borehole(rw, r, Tu, Hu, Tl, Hl, L, Kw):

    g = tf.math.log(r) - tf.math.log(rw)
    h = 1.0 + 2.0 * L * Tu / (g * rw**2 * Kw) + Tu / Tl

    ret = 2.0 * np.pi * Tu * (Hu - Hl)
    ret /= g * h

    return - ret
# %%


low = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
high = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
# %%

bounds = Bounds(lb=low, ub=high)
results = minimize_multi_start(borehole, bounds,
                               num_starts=num_starts,
                               num_samples=num_samples,
                               method="L-BFGS-B", jac=True,
                               random_state=random_state)
# %%
U = np.vstack([res.x for res in results])
v = np.hstack([res.fun for res in results])
# %%
v
# %%
res_best = min(filter(lambda res: res.success, results), key=lambda res: res.fun)
res_best
