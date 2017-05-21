"""Demonstrates how to combine datasets with different ordering

"""
import random
import string

from eelbrain import *


ds = Dataset()
# Add a Factor from a known sequence
ds['ascii'] = Factor(string.ascii_lowercase)
# Add an index variable to the dataset to later identify the cases
ds.index()

# take two separate random samples from the dataset (and print them to see
# they are shuffled)
ds1 = ds[random.sample(range(ds.n_cases), 15)]
print(ds1)
ds2 = ds[random.sample(range(ds.n_cases), 16)]
print(ds2)

# Use align to align the two datasets
ds1a, ds2a = align(ds1, ds2)

# show the ascii factors for the two datasets next to each other to
# demonstrate that they are aligned
ds1a['ascii2'] = ds2a['ascii']
print(ds1a)
