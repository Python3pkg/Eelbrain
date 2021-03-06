# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False

cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np


def l1(double [::1] x):
    cdef:
        double out = 0.
        size_t i

    for i in range(x.shape[0]):
        out += abs(x[i])

    return out


def l2(double [::1] x):
    cdef:
        double out = 0.
        size_t i

    for i in range(x.shape[0]):
        out += x[i]**2

    return out


def l1_for_delta(double [:] y, double [:] x, double delta, long shift):
    cdef:
        double out_pos = 0.
        double out_neg
        double d
        size_t i

    for i in range(shift):
        out_pos += abs(y[i])
    out_neg = out_pos

    for i in range(shift, len(y)):
        d = delta * x[i - shift]
        out_pos += abs(y[i] - d)
        out_neg += abs(y[i] + d)

    return out_pos, out_neg


def l2_for_delta(double [:] y, double [:] x, double delta, long shift):
    cdef:
        double out_pos = 0.
        double out_neg
        double d
        size_t i

    for i in range(shift):
        out_pos += (y[i]) ** 2
    out_neg = out_pos

    for i in range(shift, len(y)):
        d = delta * x[i - shift]
        out_pos += (y[i] - d) ** 2
        out_neg += (y[i] + d) ** 2

    return out_pos, out_neg


def update_error(double [::1] error, double [:] x, double delta, long shift):
    cdef:
        size_t i

    for i in range(shift, len(error)):
        error[i] -= delta * x[i - shift]
