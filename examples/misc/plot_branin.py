# -*- coding: utf-8 -*-
"""
Branin-Hoo function
===================

Hello world
"""
# sphinx_gallery_thumbnail_number = 3

import pandas as pd
import numpy as onp
import jax.numpy as np

from jax import grad, value_and_grad, vmap
from scipy.optimize import minimize, Bounds

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
# %%

y_min, y_max = 0, 15
x_min, x_max = -5, 10

y, x = onp.ogrid[y_min:y_max:200j, x_min:x_max:200j]
X, Y = np.broadcast_arrays(x, y)

x_factor = y_factor = 10

x_sparse = x[..., ::x_factor]
y_sparse = y[::y_factor]

X_sparse = X[::y_factor, ::x_factor]
Y_sparse = Y[::y_factor, ::x_factor]

seed = 42  # set random seed for reproducibility
random_state = onp.random.RandomState(seed)
# %%


def branin(x, y, a=1.0, b=5.1/(4*np.pi**2), c=5.0/np.pi, r=6.0, s=10.0,
           t=1.0/(8*np.pi)):
    return a*(y - b * x**2 + c*x - r)**2 + s*(1 - t)*np.cos(x) + s
# %%


fig, ax = plt.subplots()

ax.contour(X, Y, branin(x, y), levels=np.logspace(0, 5, 35), norm=LogNorm(),
           cmap="Spectral_r")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

plt.show()

# %%
# Branin
# ------
fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=-135, elev=35))

ax.plot_surface(x, y, branin(x, y), edgecolor='k', linewidth=0.5, cmap="Spectral_r")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")

plt.show()
# %%

dz_dx_sparse = vmap(vmap(grad(branin, argnums=0)))(X_sparse, Y_sparse)
dz_dy_sparse = vmap(vmap(grad(branin, argnums=1)))(X_sparse, Y_sparse)
# %%

fig, ax = plt.subplots()

contours = ax.pcolormesh(X, Y, branin(x, y), cmap="Spectral_r")
fig.colorbar(contours, ax=ax)

ax.quiver(x_sparse, y_sparse, -dz_dx_sparse, -dz_dy_sparse)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

plt.show()
# %%

func = value_and_grad(lambda args: branin(*args))
x0 = random_state.uniform(low=[2., 12.], high=[4., 14.])

# %%
hist = [x0]

res = minimize(func, x0=x0, method="Newton-CG",
               jac=True, tol=1e-20, callback=lambda x: hist.append(x))
print(res)
# %%

hist_arr = np.vstack(hist).T
hist_arr
# %%

fig, ax = plt.subplots()

ax.quiver(hist_arr[0, :-1], hist_arr[1, :-1],
          hist_arr[0, 1:] - hist_arr[0, :-1],
          hist_arr[1, 1:] - hist_arr[1, :-1],
          scale_units='xy', angles='xy', scale=1.0, width=3e-3, color='r')
ax.quiver(x_sparse, y_sparse, -dz_dx_sparse, -dz_dy_sparse)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

plt.show()
# %%


def line(x, y, x_lim, y_lim, step_fn, num_steps, step_size=1.0):

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    if not (num_steps and x_min <= x < x_max and y_min <= y < y_max):
        return [], []

    dx, dy = step_fn(x, y, step_size=step_size)

    xs, ys = line(x=x+dx, y=y+dy, x_lim=x_lim, y_lim=y_lim, step_fn=step_fn,
                  num_steps=num_steps-1, step_size=step_size)
    xs.append(x)
    ys.append(y)

    return xs, ys
# %%


def step(x, y, step_size):

    dx = step_size * grad(branin, argnums=0)(x, y)
    dy = step_size * grad(branin, argnums=1)(x, y)

    return dx, dy
# %%


step_sizes = 0.01 * (np.arange(10) + 1.0)
# step_size = 0.1

num_lines = 500
num_steps = 50

dfs = []

for step_size in step_sizes:

    for lineno in range(num_lines):

        # x0 = random_state.uniform(low=[x_min, y_min], high=[x_max, y_max])

        # hist = [x0]
        # res = minimize(func, x0=x0, method="TNC",
        #                bounds=[(x_min, x_max), (y_min, y_max)],
        #                jac=True, tol=1e-20, callback=lambda x: hist.append(x))
        # hist_arr = np.vstack(hist).T

        # df = pd.DataFrame(dict(lineno=lineno, x=hist_arr[0], y=hist_arr[1]))

        x, y = random_state.uniform(low=[x_min, y_min], high=[x_max, y_max])
        xs, ys = line(x, y, x_lim=(x_min, x_max), y_lim=(y_min, y_max),
                      step_fn=step, num_steps=num_steps, step_size=step_size)

        df = pd.DataFrame(dict(lineno=lineno, stepsize=step_size, x=xs, y=ys))
        dfs.append(df)
# %%

data = pd.concat(dfs, axis="index", sort=True)
data
# %%

fig, ax = plt.subplots(figsize=(10, 8))

sns.lineplot(x='x', y='y', hue='stepsize', units='lineno', estimator=None,
             sort=False, palette='Spectral', legend=None, linewidth=1.0,
             alpha=0.4, data=data, ax=ax)

plt.show()
