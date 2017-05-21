import numpy as np
from eelbrain import *
import eelbrain
from eelbrain._stats import opt

ds = datasets.get_uts(True)
y = ds['utsnd'].x
n_cases = len(y)
n_tests = np.product(y.shape[1:])
y = y.reshape((n_cases, n_tests))
m = ds.eval("A*B*rm")
fmap = np.empty((3, n_tests))
e_ms = eelbrain._stats.glm._hopkins_ems_array(m)

print("timeit -n1000 opt.anova_full_fmaps(y, m.full, m.xsinv, fmap, m._effect_to_beta, e_ms)")
