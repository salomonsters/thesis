import pickle
import numpy as np
import datetime
import numba
from numba import cuda
import geopandas
import math

def log(m):
    print("{time}: {0}".format(m, time=datetime.datetime.now()))

@numba.jit
def scale_and_average_df(df_to_scale, n_data_points, fields=('x', 'y', 'alt'), dtype=None):
    if len(df_to_scale) < n_data_points:
        # We cannot scale and average in this case, so return None
        return None
    if dtype is None:
        dtype = df_to_scale[fields[0]].dtype

    out = np.empty((n_data_points, len(fields)), dtype=dtype)
    index_new = np.round(np.linspace(0, df_to_scale.shape[0] - 1, n_data_points)).astype('int')

    df_to_scale_reindex = df_to_scale.reset_index().reindex(index_new)

    for i, field in enumerate(fields):
        out[:,i] = df_to_scale_reindex[field]
    return out

    # ts_min, ts_max = df['ts'].min(), df['ts'].max()
    # ts_len = ts_max - ts_min
    # ts_scaled = (df['ts'] - ts_min)/ts_len*n_data_points
    # df['timestamp'] = ts_scaled.apply(lambda x: pd.Timestamp(x, unit='s'))
    # df.set_index('timestamp', inplace=True)
    # resampled = df.resample(rule='10s')[fields].mean()
    # return resampled


def scale_and_average_df_multi(out_dict, k, v, *args, **kwargs):
    result = scale_and_average_df(v, *args, **kwargs)
    if result is not None:
        out_dict[k] = result

@numba.jit('void(float64[:,:],float64[:,:,:],float64)', parallel=True)
def adjacency_matrix_numba(W, x, sigma):
    for i in numba.prange(W.shape[0]):
        for j in numba.prange(W.shape[0]):
            if i == j:
                # Value on diagonal is always 1
                W[i, j] = 1.
            # We copy everything below the diagonal to above
            elif not (i < j):
                dist_squared = 0.
                for k in range(x.shape[1]):
                    for l in range(x.shape[2]):
                        dist_squared = dist_squared + (x[i, k, l] - x[j, k, l]) * (x[i, k, l] - x[j, k, l])
                W[i, j] = math.exp(-dist_squared / (2. * (sigma * sigma)))
                W[j, i] = W[i, j]
        # x_ij = x[i, :, :] - x[j, :, :]
        # distance_square = np.sum(x_ij * x_ij, axis=1)
        # W[i, j] = np.exp(np.nansum(-distance_square/ (2. * (sigma * sigma))))
        # W[j, i] = W[i, j]


def adjacency_matrix_nonumba(W, x, indices, sigma):
    for k in range(indices.shape[0]):
        i, j = indices[k]
        x_ij = x[i, :, :] - x[j, :, :]
        distance_square = np.sum(x_ij * x_ij, axis=1)
        W[i, j] = np.exp(np.nansum(-distance_square/ (2. * (sigma * sigma))))
        W[j, i] = W[i, j]


def adjacency_matrix_cuda_wrapper(W, x, indices, sigma):
    log("Start copying to device")
    # W = W[:512,:512]
    d_W = numba.cuda.device_array_like(W)
    d_x = numba.cuda.to_device(x)
    d_sigma = numba.cuda.to_device(sigma)
    log("Finished copying to device, starting calculations")
    blockdim = 16, 16
    n = W.shape[0]
    griddim = n//blockdim[0]+1, n//blockdim[1]+1
    adjacency_matrix_cuda[griddim, blockdim](d_W, d_x, d_sigma)
    numba.cuda.synchronize()
    # log("Finished calculations, copying back to device")
    d_W.copy_to_host(W)
    log("Function finished, W calculated")
    return W


@numba.cuda.jit('void(float64[:,:],float64[:,:,:],float64[:])')
def adjacency_matrix_cuda(W, x, sigma):
    i, j = numba.cuda.grid(2)

    if i < W.shape[0] and j < W.shape[0]:
        if i == j:
            # Value on diagonal is always 1
            W[i, j] = 1.
        # We copy everything below the diagonal to above
        elif not (i < j):
            dist_squared = 0.
            for k in range(x.shape[1]):
                for l in range(x.shape[2]):
                    dist_squared = dist_squared + (x[i, k, l] - x[j, k, l]) * (x[i, k, l] - x[j, k, l])
            W[i, j] = math.exp(-dist_squared / (2. * (sigma[0] * sigma[0])))
            W[j, i] = W[i, j]

import multiprocessing

def parallelize_dict(d, func, *args, **kwargs):
    manager = multiprocessing.Manager()
    out_dict = manager.dict()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = [pool.apply_async(func, args=(out_dict, k, v, *args), kwds=kwargs) for k, v in d.items()]
        _ = [r.get() for r in results]
    return out_dict

@numba.njit
def sand(box):
    out = np.array([b1 := 2*box, 10*b1])
    return out

if __name__ == "__main__":
    print(sand(5))
    # gdf = geopandas.GeoDataFrame.from_csv('data/adsb_in_eham_15.csv.gz')
    # from osgeo import osr
    #
    # inp = osr.SpatialReference()
    # inp.ImportFromEPSG(2263)
    # out = osr.SpatialReference()
    # out.ImportFromEPSG(4326)
    #
    # n_data_points = 100
    # fields = ['x', 'y']
    # one_matrix_shape = n_data_points, len(fields)
    # #
    # # df_g = tuple(gdf.groupby('fid'))
    # #
    # # data_points_per_group = np.array([len(v) for k, v in df_g])
    # #
    # # # num_groups = len(df_g)
    # # log("Completed groupby")
    # # # fid_list = np.empty(num_groups, dtype='U10')
    # #
    # # processed_dfs = parallelize_dict({fid_name: fid_df for fid_name, fid_df in df_g}, scale_and_average_df_multi,
    # #                                  n_data_points=n_data_points, fields=fields)
    # # n = len(processed_dfs)
    # # one_matrix_rows, one_matrix_cols = one_matrix_shape
    # # lower_indices = np.tril_indices(n, -1)
    # # lower_indices_list = np.array(list(zip(*lower_indices)))
    # # processed_dfs_list = list(processed_dfs.values())
    # # x_shape = tuple((len(processed_dfs_list), *one_matrix_shape))
    # # x = np.zeros(x_shape, dtype='float64')
    # # for i in range(x.shape[0]):
    # #     x[i, :, :] = processed_dfs_list[i]
    # #
    # # with open('x.pkl', 'wb') as f:
    # #     pickle.dump(x, f)
    # # with open('W_numba.pkl', 'rb') as f:
    # #     W_numba = pickle.load(f)
    # with open('x.pkl', 'rb') as f:
    #     x = pickle.load(f)
    # n = x.shape[0]
    #
    # # with open(in_fn, 'rb') as f:
    # #     processed_dfs = pickle.load(f)
    # #     fid_list = pickle.load(f)
    # #     x = pickle.load(f)
    # #     lower_indices_list = pickle.load(f)
    # #     fid_to_cluster_map = pickle.load(f)
    # log("Opened pickle")
    # sigma = 4000.
    # # lower_indices = np.tril_indices(n, -1)
    # # lower_indices_list = np.array(list(zip(*lower_indices)))
    # # indices = lower_indices_list
    #
    # W = np.ones((n,n), dtype='float64')
    # # W_nonumba = np.ones_like(W)
    # W_numba = np.ones_like(W)
    # W_cuda = np.ones_like(W)
    #
    # log("Starting numba")
    # adjacency_matrix_numba(W_numba, x, sigma)
    # log("Finished numba")
    # log("Starting CUDA")
    # W_cuda = adjacency_matrix_cuda_wrapper(W_cuda, x, None, sigma)
    # log("Finished CUDA")
    # # log("Match? {0}".format(np.allclose(W_cuda, W_numba)))
    # # log("Starting no numba")
    # # W_nonumba = np.ones_like(W)
    # # adjacency_matrix_nonumba(W_nonumba, x, indices, sigma)
    # # log("Finished no numba")
    # # log("Match? {0}".format(np.allclose(W_nonumba, W_numba)))