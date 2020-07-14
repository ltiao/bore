# -*- coding: utf-8 -*-
"""
Honey Bee Dance Data
====================

Hello world
"""
# sphinx_gallery_thumbnail_number = 7

import numpy as np
import seaborn as sns

from etudes.datasets import load_bee_dance_dataframe

golden_ratio = 0.5 * (1 + np.sqrt(5))

# %%

data = load_bee_dance_dataframe(base_dir="../../datasets")
data = data.assign(label=data.label.map(dict(waggle="waggle",
                                             turn_right="turn right",
                                             turn_left="turn left")))
# %%

g = sns.relplot(x='x', y='y', units='phase', estimator=None, hue="label",
                col="sequence", col_wrap=3, kind="line", sort=False,
                data=data, height=5, aspect=1, alpha=0.8,
                facet_kws=dict(sharex=False, sharey=False))
g.set_axis_labels(r"$x$", r"$y$")
# %%

sns.relplot(x="timestamp", y="x", units='phase', estimator=None, hue="label",
            col="sequence", col_wrap=3, kind="line",
            data=data, height=5, aspect=golden_ratio,
            alpha=0.8, facet_kws=dict(sharex=False, sharey=False))
# %%

sns.relplot(x="timestamp", y="y", units='phase', estimator=None, hue="label",
            col="sequence", col_wrap=3, kind="line",
            data=data, height=5, aspect=golden_ratio,
            alpha=0.8, facet_kws=dict(sharex=False, sharey=False))
# %%

sns.relplot(x="timestamp", y="t", units='phase', estimator=None, hue="label",
            col="sequence", col_wrap=3, kind="line",
            data=data, height=5, aspect=golden_ratio,
            alpha=0.8, facet_kws=dict(sharex=False))
# %%

long_data = data.melt(id_vars=["sequence", "timestamp", "label"],
                      value_vars=['x', 'y'], var_name="signal")

sns.relplot(x="timestamp", y="value", hue="signal",
            col="sequence", col_wrap=3, kind="line",
            data=long_data, height=5, aspect=golden_ratio,
            alpha=0.8, facet_kws=dict(sharex=False, sharey=False))
# %%

g = sns.FacetGrid(data, hue="label", col="sequence", col_wrap=3,
                  height=5, aspect=golden_ratio,
                  subplot_kws=dict(projection='polar'),
                  sharey=False, despine=False)
g.map(sns.scatterplot, "t", "timestamp")

# %%

g = sns.relplot(x="t", y="timestamp", units='phase', estimator=None, hue="label",
                col="sequence", col_wrap=3, kind="line", sort=False,
                data=data, height=5, aspect=golden_ratio,
                alpha=0.8, facet_kws=dict(subplot_kws=dict(projection='polar'),
                                          sharey=False, despine=False))
