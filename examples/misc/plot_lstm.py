# -*- coding: utf-8 -*-
"""
LSTM
====

Hello world
"""
# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import LSTM  # , Bidirectional
# %%

num_seqs = 5
seq_len = 25
num_features = 1

seed = 42  # set random seed for reproducibility
random_state = np.random.RandomState(seed)
# %%

# random walks in Euclidean space
inputs = np.cumsum(random_state.randn(num_seqs, seq_len, num_features), axis=1)
# lstm = Bidirectional(LSTM(units=1, return_sequences=True), merge_mode="concat")
lstm = LSTM(units=1, return_sequences=True)
output = lstm(inputs)
# %%

print(output.shape)
# %%

fig, ax = plt.subplots()

ax.plot(inputs[..., 0].T)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x(t)$")

plt.show()
# %%

fig, ax = plt.subplots()

ax.plot(output.numpy()[..., 0].T)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$h(t)$")

plt.show()
# %%

fig, ax = plt.subplots()

ax.plot(inputs[..., 0].T,
        output.numpy()[..., 0].T)

ax.set_xlabel(r"$x(t)$")
ax.set_ylabel(r"$h(t)$")

plt.show()
