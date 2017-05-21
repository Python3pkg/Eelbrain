# skip test (requires LaTeX)
"""Performs ANOVA and creates a PDF of the ANOVA table.

Generating PDFs requires the ``tex`` Python module to be installed.

"""
import numpy as np
from eelbrain import *

y = np.array([7, 3, 6, 6, 5, 8, 6, 7,
              7, 11, 9, 11, 10, 10, 11, 11,
              8, 14, 10, 11, 12, 10, 11, 12,
              16, 7, 11, 9, 10, 11, 8, 8,
              16, 10, 13, 10, 10, 14, 11, 12,
              24, 29, 10, 22, 25, 28, 22, 24])

a = Factor([1, 0], repeat=3 * 8, name='A')
b = Factor(list(range(3)), tile=2, repeat=8, name='B')

# Independent measures ANOVA:
subject = Factor(list(range(8 * 6)), name='subject', random=True)
anova_table = test.anova(y, a * b + subject(a % b), title="Independent Measures")

# Save the table as pdf
fmtxt.save_pdf(anova_table, 'table.pdf')

# Copy pdf to clipboard:
fmtxt.copy_pdf(anova_table)
