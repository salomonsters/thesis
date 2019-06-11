import pickle
import numpy as np
import datetime
import numba
from numba import cuda


def log(m):
    print("{time}: {0}".format(m, time=datetime.datetime.now()))


@numba.jit
def adjacency_matrix(W_flat, x_i, x_j, n, sigma, matrix_n_rows):
    x = x_i - x_j
    distance_square = np.sum(x*x, axis=1)
    distance_square_divided_negative = -distance_square / (2. * (sigma * sigma))
    for i in range(n):
        W_flat[i] = np.exp(np.nansum(distance_square_divided_negative[i*matrix_n_rows:(i+1)*matrix_n_rows]))


# @numba.jit
def preallocate_and_fill_matrices(adjacency_matrices_to_calculate, x_left, x_right, idx, one_matrix_shape):
    # log("Pre-allocated matrices, populating")

    for i, (idx_i, (x_left_i, x_right_i)) in enumerate(adjacency_matrices_to_calculate):
        assert x_left_i.shape == one_matrix_shape
        assert x_right_i.shape == one_matrix_shape
        x_left[i * one_matrix_rows:(i + 1) * one_matrix_rows, :] = x_left_i
        x_right[i * one_matrix_rows:(i + 1) * one_matrix_rows, :] = x_right_i
        idx[i] = idx_i
    #

    return x_left, x_right, idx, one_matrix_rows


log("Opening pickle")
with open('data/adjacency_matrices_to_calculate.pkl', 'rb') as f:
    adjacency_matrices_to_calculate = pickle.load(f)
log("Opened pickle")
log("Pre-allocating matrices")
#

n = len(adjacency_matrices_to_calculate)
idx = np.empty(n, dtype=tuple)
one_matrix_shape = adjacency_matrices_to_calculate[0][1][0].shape
one_matrix_rows, one_matrix_cols = one_matrix_shape
shape = n * one_matrix_shape[0], one_matrix_shape[1]
x_left = np.zeros(shape, dtype='float64')
x_right = np.zeros(shape, dtype='float64')

x_left, x_right, idx, one_matrix_rows = preallocate_and_fill_matrices(adjacency_matrices_to_calculate, x_left, x_right, idx, one_matrix_shape)
log("Populated matrices")
W_shape = [1+i for i in np.max(idx)]
W = np.zeros(W_shape, dtype='float64')
W_flat = W.ravel()
adjacency_matrix(W_flat, x_left, x_right, len(idx), 26561*.3, one_matrix_rows)
W = W_flat.reshape(W_shape)

log("Calculated adjacency matrix")
