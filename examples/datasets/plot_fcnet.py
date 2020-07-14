# -*- coding: utf-8 -*-
"""
FCNet Tabular Benchmarks
========================

Hello world
"""
# sphinx_gallery_thumbnail_number = 1

import seaborn as sns
import h5py

from etudes.datasets import read_fcnet_data
# %%

name = "protein_structure"
path = f"../../datasets/fcnet_tabular_benchmarks/fcnet_{name}_data.hdf5"

with h5py.File(path, 'r') as f:
    data = read_fcnet_data(f, max_configs=500)
# %%

a = data.query("batch_size == 16 and "
               "activation_fn_1 == 'relu' and "
               "activation_fn_2 == 'relu' and "
               "init_lr == 1e-3 and "
               "lr_schedule == 'cosine'")

g = sns.relplot(x="epoch", y="valid_mse",
                hue="n_units_1", style="n_units_2",
                row="dropout_1", col="dropout_2", ci="sd",
                # units="seed", estimator=None,
                kind="line", palette="colorblind", data=a)
g.set(xscale="log", yscale="log")
