from eelbrain import *
import numpy as np

y = np.empty(21)
y[:14] = np.random.normal(0, 1, 14)
y[14:] = np.random.normal(1.5, 1, 7)

Y = Var(y, 'Y')
A = Factor('abc', 'A', repeat=7)

print(Dataset((Y, A)))
print(table.frequencies(A))
print(test.anova(Y, A))
print(test.pairwise(Y, A, corr='Hochberg'))

t = test.pairwise(Y, A, corr='Hochberg')
print(t.get_tex())

plot.Boxplot(Y, A, title="My Boxplot", ylabel="value", corr='Hochberg')
