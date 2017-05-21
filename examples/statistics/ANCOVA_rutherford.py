"""Univariate ANCOOVA

results cross-checked with Rutherford (2001), pp. 118-20

Results according to Rutherford (2001, p. 120)
(without interaction term)

         	  SS     	df 	  MS     	    F      	 p
---------------------------------------------------------
A         	807.8161 	 2 	403.9080 	62.8830*** 	.0000
Covariate 	199.5366 	 1 	199.5366 	31.0651*** 	.0000
Residuals 	128.4634 	20 	  6.4232
---------------------------------------------------------
Total     	    1112 	23

"""
import numpy as np
from eelbrain import *


Y = np.array([16,  7, 11,  9, 10, 11,  8,  8,
              16, 10, 13, 10, 10, 14, 11, 12,
              24, 29, 10, 22, 25, 28, 22, 24])

cov = Var([9, 5, 6, 4, 6, 8, 3, 5, 
           8, 5, 6, 5, 3, 6, 4, 6,
           5, 8, 3, 4, 6, 9, 4, 5], name='cov')

A = Factor([1, 2, 3], repeat=8, name='A')

print(test.anova(Y, A + cov))
print(test.anova(Y, cov * A))
plot.Regression(Y, cov)
plot.Regression(Y, cov, A)
