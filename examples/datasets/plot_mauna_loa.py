# -*- coding: utf-8 -*-
"""
Mauna Loa Atmospheric Carbon Dioxide
====================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 1

import seaborn as sns

from etudes.datasets import mauna_loa_load_dataframe

# %%

data = mauna_loa_load_dataframe(base_dir="../../datasets")

g = sns.relplot(x='date', y='average', kind="line",
                data=data, height=5, aspect=1.5, alpha=0.8)
g.set_ylabels(r"average $\mathrm{CO}_2$ (ppm)")
