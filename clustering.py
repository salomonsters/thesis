import copy
from itertools import cycle

import geopandas
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from numba import cuda
from scipy.linalg import eigh
import contextily as ctx

from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
from tools import create_logger


def adjacency_matrix_cuda_wrapper(W, x, sigma):
    sigma = np.array(sigma, dtype='float64')
    # log("Start copying to device")
    d_W = numba.cuda.device_array_like(W)
    d_x = numba.cuda.to_device(x)
    d_sigma = numba.cuda.to_device(sigma)
    # log("Finished copying to device, starting calculations")
    blockdim = 16, 16
    n = W.shape[0]
    griddim = n//blockdim[0]+1, n//blockdim[1]+1
    adjacency_matrix_cuda[griddim, blockdim](d_W, d_x, d_sigma)
    d_W.copy_to_host(W)
    # log("Function finished, W calculated")


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


@numba.jit('void(float64[:,:],float64[:,:,:],int64)', parallel=True)
def adjacency_matrix_numba(W, x, K):
    D = np.zeros_like(W)
    sigmas = np.zeros(W.shape[0], dtype=D.dtype)
    distance_matrix_numba(D, x, sigmas, K)
    for i in numba.prange(W.shape[0]):
        for j in numba.prange(W.shape[0]):
            if i == j:
                # Value on diagonal is always 0
                W[i, j] = 0.
            # We copy everything below the diagonal to above
            elif not (i < j):
                dist_squared = D[i, j]
                sigma_i_j_product = sigmas[i] * sigmas[j]
                W[i, j] = math.exp(-dist_squared / (2. * sigma_i_j_product))
                W[j, i] = W[i, j]


@numba.jit('void(float64[:,:],float64[:,:,:],float64[:],int64)', parallel=True)
def distance_matrix_numba(D, x, sigmas, K):
    for i in numba.prange(D.shape[0]):
        for j in numba.prange(D.shape[0]):
            if i == j:
                # Value on diagonal is always 0
                D[i, j] = 0.
            # We copy everything below the diagonal to above
            elif not (i < j):
                dist_squared = 0.
                for k in range(x.shape[1]):
                    for l in range(x.shape[2]):
                        dist_squared = dist_squared + (x[i, k, l] - x[j, k, l]) * (x[i, k, l] - x[j, k, l])
                D[i, j] = dist_squared
                D[j, i] = dist_squared
    for i in numba.prange(D.shape[0]):
        j = np.argsort(D[i, :])[K - 1]
        sigmas[i] = np.sqrt(D[i, j])


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
    for i in numba.prange(ary_out.shape[0]):
        na_indices = np.where(np.isnan(ary_in[i]))[0]
        if na_indices.shape[0] == 0:
            first_na_index = ary_in[i].shape[0]
        else:
            first_na_index = na_indices[0]
        index_new = np.round(np.linspace(0, first_na_index - 1, n_data_points)).astype('int')
        ary_out[i, :, :] = ary_in[i, index_new, :]


def scale_and_average_df_numba_wrapper(df, sample_to_n_rows, fields=('lat', 'lon'), dtype='float64'):
    for field_counter, field in enumerate(fields):
        fields = list(fields)
        if field == 'lat':
            df['x'] = lon2x(df['lon'].values)
            fields[field_counter] = 'x'
        if field == 'lon':
            df['y'] = lat2y(df['lat'].values)
            fields[field_counter] = 'y'
        if field == 'alt':
            df['alt_scaled'] = 3*1852*0.3048*df['alt']/2
            fields[field_counter] = 'alt_scaled'

    df_grouped = df.set_index('fid')[fields].groupby('fid')
    index_map_naive = np.array(df_grouped.count().index, dtype='str')
    gdf_as_numpy_arrays = df_grouped.apply(pd.DataFrame.to_numpy)
    rows_per_numpy_array = [_.shape[0] for _ in gdf_as_numpy_arrays]
    converted_df = np.array([gdf_as_numpy_arrays[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows >= sample_to_n_rows])
    discarded_fids = np.array([index_map_naive[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows < sample_to_n_rows])
    index_map = np.array([index_map_naive[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows >= sample_to_n_rows])
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


def adjacency_matrix(W, x, sigma, use_cuda=False, K=None):
    if use_cuda:
        if K:
            raise NotImplementedError("Setting K is not implemented for Cuda yet")
        adjacency_matrix_cuda_wrapper(W, x, sigma)
    else:
        adjacency_matrix_numba(W, x, K)


def fiedler_vector(L):
    l, U = eigh(L)
    f = U[:, 1]
    return f


@numba.njit
def select_rows_and_columns(matrix, rows, columns):
    out = np.zeros((len(rows), len(columns)), dtype=matrix.dtype)
    for i in numba.prange(len(rows)):
        for j in numba.prange(len(columns)):
            out[i, j] = matrix[rows[i], columns[j]]
    return out


@numba.jit(nopython=True)
def array_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True

# @numba.njit(parallel=False)
def spectralCluster(W, tresholds, result_indices, original_indices, min_cluster_size):
    e_mean, e_var = tresholds

    # stop_function = lambda X, Y: np.var(X)/np.var(Y) < omega_min
    # stop_function = lambda X, Y: np.max(X) / np.max(Y) < 4*omega_min
    stop_function = lambda X, Y: np.var(X)/np.var(Y) < e_var or np.mean(X) > e_mean
    # stop_function = lambda X, Y: np.std(X) < omega_min

    D = np.zeros_like(W)
    for i in range(W.shape[0]):
        D[i,i] = np.sum(W[i,:])
    L = D - W
    # Fiedler vector:
    # v = fiedler_vector(L)
    l, U = np.linalg.eigh(L)

    v = U[:, 1]

    # Indices of V with positive elements
    i_l = np.where(v >= 0)[0]
    i_r = np.where(v < 0)[0]
    if len(i_r) > len(i_l):
        # we want to go ahead with the biggest partition first
        i_l, i_r = i_r, i_l
    W_il_il = select_rows_and_columns(W, i_l, i_l)
    W_ir_ir = select_rows_and_columns(W, i_r, i_r)

    # plot_cluster(x[original_indices[i_l]].reshape((-1, x.shape[2])), x[original_indices[i_r]], airspace_projected,
    #              "blue: n={0}, mean={2:.4f}, var={4:.4f}; orange: n={1}, mean={3:.4f}, var={5:.4f}".format(len(i_l),
    #                                                                                                        len(i_r),
    #                                                                                                        np.mean(W_il_il),
    #                                                                                                        np.mean(W_ir_ir),
    #                                                                                                        np.var(W_il_il)/np.var(W),
    #                                                                                                        np.var(W_ir_ir)/np.var(W)))

    # Stop either when the stop function is reached, or when we are not partitioning anymore
    if len(i_l) == 0 or len(i_r) == 0:
        if len(i_l) == 0:
            result_indices[original_indices[i_r]] = -1
            # result_indices[original_indices[i_r]] = np.max(result_indices) + 1
        if len(i_r) == 0:
            result_indices[original_indices[i_l]] = -1
            # result_indices[original_indices[i_l]] = np.max(result_indices) + 1
    else:
        if stop_function(W_il_il, W) or array_equal(original_indices[i_l], original_indices):  # or len(i_l) < min_cluster_size:
            result_indices[original_indices[i_l]] = np.max(result_indices) + 1
        else:
            spectralCluster(W_il_il, tresholds, result_indices, original_indices[i_l], min_cluster_size)
        if stop_function(W_ir_ir, W) or array_equal(original_indices[i_r], original_indices):  # or len(i_r) < min_cluster_size:
            result_indices[original_indices[i_r]] = np.max(result_indices) + 1
        else:
            spectralCluster(W_ir_ir, tresholds, result_indices, original_indices[i_r], min_cluster_size)


def inverse_map(map):
    inv_map = {}
    for k, v in map.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map


def plot_cluster(tracks_mean, tracks_concat, airspace_projected, title):
    tracks_concat_flat = tracks_concat.reshape((-1, x.shape[2]))

    fig = plt.figure()

    ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

    ax.set_axis_off()
    colorcycle = cycle(['C0', 'C1'])
    sizecycle = cycle([1, 0.1])
    for tracks in [tracks_mean, tracks_concat_flat]:
        color = next(colorcycle)
        size = next(sizecycle)
        gs = geopandas.GeoSeries(geopandas.points_from_xy(tracks[:, 0], tracks[:, 1]))
        gs.crs = {'init': 'epsg:3857', 'no_defs': True}
        gs.plot(ax=ax, color=color, markersize=size, linewidth=size)
    # ax.scatter(tracks_mean[:, 0].min() - 1.2 * (tracks_mean[:, 0].max() - tracks_mean[:, 0].min()),
    #            tracks_mean[:, 1].min(), c='C2')

    plt.title(title)
    plt.show()

def plot_means(tracks, unclustered, airspace):
    fig = plt.figure()
    airspace_projected = prepare_gdf_for_plotting(airspace)
    ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

    ax.set_axis_off()
    gs = geopandas.GeoSeries(geopandas.points_from_xy(tracks[:, 0], tracks[:, 1]))
    gs.crs = {'init': 'epsg:3857', 'no_defs': True}
    gs.plot(ax=ax, markersize=1, linewidth=1)
    # gs_unclustered = geopandas.GeoSeries(geopandas.points_from_xy(unclustered[:, 0], unclustered[:, 1]))
    # gs_unclustered.crs = {'init': 'epsg:3857', 'no_defs': True}
    # gs_unclustered.plot(ax=ax, markersize=0.1, linewidth=0.1)
    plt.show()


if __name__ == "__main__":
    verbose = True
    airspace_query = "airport=='EHAM'"
    zoom = 12
    minalt = 200  # ft
    maxalt = 10000
    sigma = 5000000.
    e_mean = 0.4
    e_var = 1
    min_cluster_size = 10
    n_data_points = 200
    fields = ['lat', 'lon']
    K = 7
    use_cuda = False
    plot_individual_clusters = True
# todo write verification cases for spectralcluster to see if partitioning is handled properly
# todo maybe check some cases where we visually see overlap?
    one_matrix_shape = n_data_points, len(fields)
    airspace = ehaa_airspace.query(airspace_query)
    # stop = lambda X, Y: np.max(X) / np.max(Y) < omega_min
    airspace_projected = prepare_gdf_for_plotting(airspace)
    machine_precision = np.finfo(np.float64).eps


    log = create_logger(verbose, "Clustering")


    log("Start reading csv")
   #fids = ['01cf3c72', '03adb898', '056b3610', '05df1fbc', '08f137f8', '0b434230', '0efd9cb8', '14bcd650', '1d59b4fe', '21235a40', '22b871c4', '241eedfe', '250a569a', '26f4585c', '2c1516b4', '2c587800', '332580a6', '33946408', '3b997dfa', '3c676710', '42571de6', '45ec369e', '4b478e5e', '4e45152c', '4f2c2340', '505ba72c', '52b82b12', '53f0df2e', '56849a00', '56e809f0', '5c019d98', '608934e8', '6178a582', '623ac482', '6511a658', '65267330', '669a0056', '6d4d91f6', '6f17fc60', '7155f360', '71720abe', '74b08d5e', '76e15874', '7ba88512', '7dab8698', '7ec98606', '816bd666', '82422cc0', '8826b82c', '89066a80', '8c1f1db6', '8d3988e4', '8e7d7bc0', '90899a8e', '950db6d0', '9afb2cda', '9b32d590', '9d9dcc36', '9df2dcd0', 'a189653a', 'a26144dc', 'a2bf10b2', 'a2e25932', 'a39c57ec', 'a89c31ae', 'a97caffe', 'a9fb519c', 'aa574312', 'b18dc9d0', 'b255eda2', 'b48ff450', 'b580d01e', 'b879654c', 'bf6c4c20', 'c054a1b4', 'c1b42aac', 'c2b7226a', 'c62d1558', 'ca97e870', 'cd69d734', 'cd6cfbb2', 'ce716ed0', 'cfa4d5ee', 'd11c8368', 'd492403c', 'd510fb8e', 'd71634a8', 'd830f332', 'd882bb54', 'dd1b2c1e', 'e98a5048', 'ea55d650', 'ef9593f2', 'fb0895aa', 'ff616302']

    # #### SINGLE FILENAME
    df = pd.read_csv('data/adsb_decoded_in_eham/ADSB_DECODED_20180101.csv.gz')  # , converters={'callsign': lambda s: s.replace('_', '')})
    #df = df.query("fid in @fids")

    # #### ENTIRE DIRECTORY
    # path = './data/adsb_decoded_in_eham/'
    # max_number_of_files = 20
    # df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))[:max_number_of_files]))
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
    adjacency_matrix(W, x, sigma, use_cuda=use_cuda, K=K)
    log("Calculated adjacency matrix")

    cluster_result = np.zeros(num_groups, dtype='int')
    original_indices = np.array(range(W.shape[0]))
    spectralCluster(W, (e_mean, e_var), cluster_result, original_indices, min_cluster_size)
    log("Finished spectralCluster")
    fid_to_cluster_map = {fid_list[i]: c_r for i, c_r in enumerate(cluster_result)}
    # for cluster_number, fidlist_indices in enumerate(result):
    #     for fid in fid_list[fidlist_indices.ravel()]:
    #         fid_to_cluster_map[fid] = cluster_number
    df.query('fid not in @discarded_fids', inplace=True)
    df['cluster'] = df['fid'].map(fid_to_cluster_map)
    log("Determined fid_to_cluster_map")
    log("{0} clusters found (sigma={1}, e_mean={2}, e_var={3})".format(1+max(fid_to_cluster_map.values()), sigma, e_mean, e_var))

    cluster_to_fid_map = inverse_map(fid_to_cluster_map)
    fid_to_index_map = {v: k for k, v in enumerate(fid_list)}
    tracks_means = []
    noise = None

    # Sort from largest to smallest cluster
    for key, fids in sorted(cluster_to_fid_map.items(), key=lambda a: len(a[1]))[::-1]:
        tracks_concat_index = np.array([fid_to_index_map[fid] for fid in fids])
        tracks_concat = x[tracks_concat_index]
        if key == -1:
            noise = tracks_concat.reshape((-1, x.shape[2]))
            continue
        tracks_mean = tracks_concat.mean(axis=0)
        tracks_means.append(tracks_mean)
        if plot_individual_clusters:
            plot_cluster(tracks_mean, tracks_concat, airspace_projected, "$n={0}, \epsilon_{{mean}}={1},\epsilon_{{var}}={2}$".format(len(fids), e_mean, e_var))
            if input("Continue? [y/n]").capitalize() == "N":
                break
    plot_means(np.vstack(tracks_means), noise, airspace)
    # W_cluster = np.zeros((tracks_concat.shape[0], tracks_concat.shape[0]), dtype='float64')
    # adjacency_matrix(W_cluster, tracks_concat, sigma, use_cuda=use_cuda)

