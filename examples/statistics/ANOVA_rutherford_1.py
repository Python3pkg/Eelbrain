"""Univariate repeated measures ANOVA

Rutherford (2001) Examples, cross-checked results:

independent measure (p. 24):
3 groups, 8 subjects each

            SS   df      MS   MS(denom)   df(denom)          F        p
-----------------------------------------------------------------------
A       112.00    2   56.00        2.48          21   22.62***   < .001
-----------------------------------------------------------------------
Total   164.00   23


repeated measures (p. 72):
8 s on 3 blocks

            SS   df      MS   MS(denom)   df(denom)          F        p
-----------------------------------------------------------------------
A       112.00    2   56.00        2.71          14   20.63***   < .001
-----------------------------------------------------------------------
Total   164.00   23

"""
import numpy as np
from eelbrain import *


Y = Var([7,  3,  6,  6,  5,  8,  6,  7,
         7, 11,  9, 11, 10, 10, 11, 11,
         8, 14, 10, 11, 12, 10, 11, 12],
        name='Y')

A = Factor(8 * 'a' + 8 * 'b' + 8 * 'c', name='A')

# Independent Measures, as mixed effects model:
subject = Factor(list(range(24)), name='subject', random=True)
aim = test.anova(Y, A + subject(A), title="Independent Measures Full Model")
print(aim)
# as fixed effects model
print((test.anova(Y, A, title="Independent Measures")))


# Repeated Measures:
subject = Factor(np.array(list(range(8)) * 3), name='subject', random=True)
arm = test.anova(Y, A * subject, title="Repeated Measures")
print(arm)
