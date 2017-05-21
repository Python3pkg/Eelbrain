"""Univariate 2-way ANOVA
"""
from eelbrain import *

ds = datasets.get_uv()

print(test.anova('fltvar', 'A*B', ds=ds))

plot.Boxplot('fltvar', 'A%B', ds=ds)
