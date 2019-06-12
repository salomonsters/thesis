import numpy as np
import copy
import math
from itertools import cycle
from scipy.linalg import eigh
import pandas as pd
import geopandas
from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import datetime
import numba
from numba import cuda
import glob, os


def log(m):
    print("{time}: {0}".format(m, time=datetime.datetime.now()))


def adjacency_matrix_cuda_wrapper(W, x, sigma):
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
    # numba.cuda.synchronize()
    # log("Finished calculations, copying back to device")
    d_W.copy_to_host(W)
    log("Function finished, W calculated")


@numba.cuda.jit('void(float64[:,:],float64[:,:,:],float64[:])')
def adjacency_matrix_cuda(W, x, sigma):
    i, j = numba.cuda.grid(2)

    if i < W.shape[0] and j < W.shape[0]:
        if i == j:
            # Value on diagonal is always 1
            W[i, j] = 1.
        # We copy everything below the diagonal to above
        elif not (i < j):
            W[i, j] = 0.1
            dist_squared = 0.
            for k in range(x.shape[1]):
                for l in range(x.shape[2]):
                    dist_squared = dist_squared + (x[i, k, l] - x[j, k, l]) * (x[i, k, l] - x[j, k, l])
            W[i, j] = math.exp(-dist_squared / (2. * (sigma[0] * sigma[0])))
            W[j, i] = W[i, j]


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


@numba.vectorize('float64(float64)')
def lat2y(lat):
    return math.log(math.tan(math.radians(lat) / 2 + math.pi/4)) * 6378137.0

@numba.vectorize('float64(float64)')
def lon2x(lon):
    return math.radians(lon) * 6378137.0

# Ary_in: first dimension is fid, second is rows, third is data
# ary out: first dimension fid, second rows, third data (but then scaled and averaged)
@numba.jit(parallel=True)
def scale_and_average_df_numba(ary_in, ary_out):
    n_data_points = ary_out.shape[1]
    for i in range(ary_out.shape[0]):
        na_indices = np.where(np.isnan(ary_in[i]))[0]
        if na_indices.shape[0] == 0:
            first_na_index = ary_in[i].shape[0]
        else:
            first_na_index = na_indices[0]
        index_new = np.round(np.linspace(0, first_na_index - 1, n_data_points)).astype('int')
        ary_out[i, :, :] = ary_in[i, index_new, :]


def scale_and_average_df_numba_wrapper(df, sample_to_n_rows, fields=['lat', 'lon'], dtype='float64'):
    for field_counter, field in enumerate(fields):
        if field == 'lat':
            df['x'] = lon2x(df['lon'].values)
            fields[field_counter] = 'x'
        if field == 'lon':
            df['y'] = lat2y(df['lat'].values)
            fields[field_counter] = 'y'

    df_grouped = df.set_index('fid')[fields].groupby('fid')
    index_map = np.array(df_grouped.count().index, dtype='str')
    gdf_as_numpy_arrays = df_grouped.apply(pd.DataFrame.to_numpy)
    rows_per_numpy_array = [_.shape[0] for _ in gdf_as_numpy_arrays]
    converted_df = np.array([gdf_as_numpy_arrays[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows >= sample_to_n_rows])
    discarded_fids = np.array([index_map[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows < sample_to_n_rows])
    max_n_datapoints = np.max(rows_per_numpy_array)
    n_fields = len(fields)
    ary_in_shape = converted_df.shape[0], max_n_datapoints, n_fields
    ary_in = np.zeros(ary_in_shape, dtype=dtype)
    ary_in[:,:,:] = np.nan

    for i in range(converted_df.shape[0]):
        ary_to_fil = converted_df[i]
        ary_in[i, :ary_to_fil.shape[0], :] = ary_to_fil

    ary_out_shape = converted_df.shape[0], sample_to_n_rows, n_fields
    ary_out = np.zeros(ary_out_shape, dtype=dtype)
    scale_and_average_df_numba(ary_in, ary_out)

    return ary_out, index_map, discarded_fids

def adjacency_matrix(W, x, sigma, use_cuda=False):
    if use_cuda:
        adjacency_matrix_cuda_wrapper(W, x, sigma)
    else:
        adjacency_matrix_numba(W, x, sigma)

def fiedler_vector(L):
    l, U = eigh(L)
    f = U[:, 1]
    return f

@numba.njit
def spectralCluster(W, stop_function, result_indices, original_indices=None):
    if original_indices is None:
        original_indices = np.array(range(W.shape[0]))
    D = np.zeros_like(W)
    for i in range(W.shape[0]):
        D[i,i] = np.sum(W[i,:])
    L = D - W
    v = fiedler_vector(L)
    # Indices of V with positive elements
    i_l = np.argwhere(v >= 0).ravel()
    i_r = np.argwhere(v < 0).ravel()
    W_il_il = W[np.ix_(i_l, i_l)]
    W_ir_ir = W[np.ix_(i_r, i_r)]

    # Stop either when the stop function is reached, or when we are not partitioning anymore
    if len(i_l) == 0 or len(i_r) == 0:
        if len(i_l) == 0:
            result_indices.append(original_indices[i_r])
        if len(i_r) == 0:
            result_indices.append(original_indices[i_l])
    else:
        if stop_function(W_il_il, W) or np.array_equal(original_indices[i_l], original_indices) or len(i_l) == 1:
            result_indices.append(original_indices[i_l])
        else:
            spectralCluster(W_il_il, stop_function, result_indices, original_indices[i_l])
        if stop_function(W_ir_ir, W) or np.array_equal(original_indices[i_r], original_indices) or len(i_r) == 1:
            result_indices.append(original_indices[i_r])
        else:
            spectralCluster(W_ir_ir, stop_function, result_indices, original_indices[i_r])

def inverse_map(map):
    inv_map = {}
    for k, v in map.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map


if __name__ == "__main__":
    airspace_query = "airport=='EHAM'"
    zoom = 15
    minalt = 200  # ft
    maxalt = 10000
    sigma = 4000.
    omega_min = 1
    n_data_points = 100
    fields = ['lat', 'lon']
    use_cuda = False

    one_matrix_shape = n_data_points, len(fields)
    airspace = ehaa_airspace.query(airspace_query)
    # stop = lambda X, Y: np.max(X) / np.max(Y) < omega_min

    stop = lambda X, Y: np.var(X)/np.var(Y) < omega_min


    log("Start reading csv")

    # #### SINGLE FILENAME
    # df = pd.read_csv('data/adsb_decoded_in_eham/ADSB_DECODED_20180101.csv.gz')  # , converters={'callsign': lambda s: s.replace('_', '')})

    # #### ENTIRE DIRECTORY
    path = './data/adsb_decoded_in_eham/'
    max_number_of_files = 15
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))[:max_number_of_files]))
    log("Completed reading CSV")

    orig_df = copy.deepcopy(df)

    if minalt is not None:
        df = df[df['alt'] > minalt]
    if maxalt is not None:
        df = df[df['alt'] < maxalt]

    x, fid_list, discarded_fids = scale_and_average_df_numba_wrapper(df, n_data_points, fields)
    log("Threw away {0} fid's that had less than {1} rows".format(len(discarded_fids), n_data_points))

    num_groups = x.shape[0]

    log("Converted coordinates, Queued adjacency matrix calculation")

    W = np.zeros((num_groups, num_groups), dtype='float64')
    # Function below modifies W in place
    adjacency_matrix(W, x, sigma, use_cuda=use_cuda)
    log("Calculated adjacency matrix")

    result = []
    spectralCluster(W, stop, result)
    log("Finished spectralCluster")
    fid_to_cluster_map = {}
    for cluster_number, fidlist_indices in enumerate(result):
        for fid in fid_list[fidlist_indices.ravel()]:
            fid_to_cluster_map[fid] = cluster_number
    df.query('fid not in @discarded_fids', inplace=True)
    df['cluster'] = df['fid'].map(fid_to_cluster_map)
    log("Determined fid_to_cluster_map")
    log("{0} clusters found (sigma={1}, omega_min={2})".format(1+max(fid_to_cluster_map.values()), 1, omega_min))

    cluster_to_fid_map = inverse_map(fid_to_cluster_map)
    fid_to_index_map = {v: k for k, v in enumerate(fid_list)}
    # Sort from largest to smallest cluster
    for key, fids in sorted(cluster_to_fid_map.items(), key=lambda a: len(a[1]))[::-1]:
        tracks_concat_index = np.array([fid_to_index_map[fid] for fid in fids])
        tracks_concat = x[tracks_concat_index]
        tracks_concat_flat = tracks_concat.reshape((-1, x.shape[2]))
        tracks_mean = tracks_concat.mean(axis=0)

        fig = plt.figure()
        airspace_projected = prepare_gdf_for_plotting(airspace)
        ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

        # add_basemap(ax, zoom=zoom, ll=False)

        ax.set_axis_off()
        colorcycle = cycle(['C0', 'C1'])
        sizecycle = cycle([1, 0.1])
        for tracks in [tracks_mean, tracks_concat_flat]:
            color = next(colorcycle)
            size = next(sizecycle)
            gs = geopandas.GeoSeries(geopandas.points_from_xy(tracks[:, 0], tracks[:, 1]))
            gs.crs = {'init': 'epsg:3857', 'no_defs': True}
            gs.plot(ax=ax, color=color, markersize=size, linewidth=size)
        plt.show()
        if input("Continue? [y/n]").capitalize() == "N":
            break