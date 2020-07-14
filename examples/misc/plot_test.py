# -*- coding: utf-8 -*-
"""
Test
====

Hello world
"""
# sphinx_gallery_thumbnail_number = 3

from tabular_benchmarks import FCNetProteinStructureBenchmark
# %%
max_epochs = 100
# %%

b = FCNetProteinStructureBenchmark(data_dir="../../datasets/fcnet_tabular_benchmarks")
cs = b.get_configuration_space()
config = cs.sample_configuration()
# %%

print("Numpy representation: ", config.get_array())
# %%

print("Dict representation: ", config.get_dictionary())
# %%

y, cost = b.objective_function(config, budget=max_epochs)
# %%

print(y, cost)
