import math
import pytest

import numpy as np
import pandas as pd

from clustering import scale_and_average_df_numba, scale_and_average_df_numba_wrapper
from clustering import lat2y, lon2x
from clustering import adjacency_matrix_numba, adjacency_matrix_cuda_wrapper

@pytest.fixture
def df_to_scale():
    n = 151
    records_fid_1 = {'ts': np.linspace(1560417691.4821393, 1560417691.4, n),
               'lat': np.linspace(52., 53., n),
               'lon': np.linspace(4., 5., n),
               'alt': np.linspace(100., 12000., n),
               'fid': 'abcdef'}

    records_fid_2 = {'ts': np.linspace(1560417691.4821393, 1560417691.4, n),
               'lat': np.linspace(51., 53., n),
               'lon': np.linspace(3., 5., n),
               'alt': np.linspace(12000., 100., n),
               'fid': 'fedcba'}

    df = pd.concat((pd.DataFrame.from_dict(records_fid_1),
                    pd.DataFrame.from_dict(records_fid_2)), axis=0).reset_index()
    return df


@pytest.fixture
def ary_scaled_expected():
    return np.array([[[4.45277963e+05, 6.80012545e+06, 1.00000000e+02],
                      [5.00937709e+05, 6.89104172e+06, 6.05000000e+03],
                      [5.56597454e+05, 6.98299792e+06, 1.20000000e+04]],
                     [[3.33958472e+05, 6.62129372e+06, 1.20000000e+04],
                      [4.45277963e+05, 6.80012545e+06, 6.05000000e+03],
                      [5.56597454e+05, 6.98299792e+06, 1.00000000e+02]]], dtype='float64')


@pytest.fixture
def sigma():
    return 100000

@pytest.fixture
def adjacency_matrix_expected(sigma):
    W = np.zeros((2, 2), dtype='float64')
    W[0, 0] = 1
    W[1, 1] = 1
    W[0, 1] = math.exp(-(
        (4.45277963e+05 - 3.33958472e+05)**2 + (5.00937709e+05 - 4.45277963e+05)**2 + (5.56597454e+05 - 5.56597454e+05)**2 +
        (6.80012545e+06 - 6.62129372e+06)**2 + (6.89104172e+06 - 6.80012545e+06)**2 + (6.98299792e+06 - 6.98299792e+06)**2 +
        (100 - 12000)**2 + (6050 - 6050)**2 + (12000 - 100)**2
                       ) / (2 * sigma * sigma))
    W[1, 0] = W[0, 1]
    return W


def test_lat2y():
    assert lat2y(52.0) == pytest.approx(6.80012545e+06)
    assert np.allclose(lat2y([52.0, 52.5]), [6.80012545e+06, 6.89104172e+06], atol=1e-1)


def test_lon2x():
    assert lon2x(4.0) == pytest.approx(4.45277963e+05)
    assert np.allclose(lon2x([4.0, 4.5]), [4.45277963e+05, 5.00937709e+05], atol=1e-1)


def test_scale_and_average_df(df_to_scale, ary_scaled_expected):
    sample_to_n_rows = 3
    original_n_rows = 151
    n_fids = 2
    fields = ['lat', 'lon']
    ary_in = np.array([df_to_scale[fields].values[:original_n_rows],
                       df_to_scale[fields].values[original_n_rows:]])
    ary_out = np.zeros((n_fids, sample_to_n_rows, len(fields)), dtype=ary_in.dtype)
    ary_out_expected = np.array([[[52.,     4.],
                                  [52.5,    4.5],
                                  [53.,     5.]],
                                 [[51., 3.],
                                  [52., 4.],
                                  [53., 5.]]])
    scale_and_average_df_numba(ary_in, ary_out)
    assert np.allclose(ary_out, ary_out_expected)

    fields = ('lat', 'lon', 'alt')
    df_to_scale_added_fid = df_to_scale.append([{'ts': 1, 'lat': 50, 'lon': 3, 'alt': 13, 'fid': 'aaaaaa'}])
    ary_scaled, index_map, discarded_fids = scale_and_average_df_numba_wrapper(df_to_scale_added_fid, sample_to_n_rows, fields=fields)
    assert ary_scaled.shape == (n_fids, sample_to_n_rows, len(fields))
    assert discarded_fids == ['aaaaaa']
    assert np.array_equal(index_map, np.array(['abcdef', 'fedcba']))
    # ary_scaled_expected = np.array([[[lon2x(4.), lat2y(52.), 100.],
    #                                  [lon2x(4.5), lat2y(52.5), 6050.],
    #                                  [lon2x(5.), lat2y(53.), 12000.]],
    #                                 [[lon2x(3.), lat2y(51.), 12000.],
    #                                  [lon2x(4.), lat2y(52.), 6050.],
    #                                  [lon2x(5.), lat2y(53.), 100.]]])

    assert np.allclose(ary_scaled, ary_scaled_expected)


def test_adjacency_matrix(ary_scaled_expected, adjacency_matrix_expected, sigma):
    W_cuda = np.zeros((2, 2), dtype='float64')
    W_numba = np.zeros((2, 2), dtype='float64')
    adjacency_matrix_cuda_wrapper(W_cuda, ary_scaled_expected, sigma)
    adjacency_matrix_numba(W_numba, ary_scaled_expected, sigma)
    assert np.allclose(W_numba, adjacency_matrix_expected)
    # assert np.allclose(W_cuda, adjacency_matrix_expected)
