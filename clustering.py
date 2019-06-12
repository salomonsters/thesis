import numpy as np
import copy
import scipy as sp
import math
from itertools import cycle
from scipy.linalg import eigh
import pandas as pd
import itertools
import scipy as sp
from scipy import signal
import geopandas
from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import datetime
import glob
import os
import multiprocessing
import mpld3
from mpld3 import plugins
import numba
from numba import cuda

def log(m):
    print("{time}: {0}".format(m, time=datetime.datetime.now()))

# TODO: then maybe pad as well so we don't get nan's?
# TODO: perhaps normalize to a track with a specific airspeed so we have more comparable tracks
# and then don't group on length?
#
# def adjacency_matrix(x_i, x_j, sigma):
#     x = x_i - x_j
#     x.dropna(inplace=True)
#     if len(x) == 0:
#         return 0
#     dist = np.linalg.norm(x)
#     return np.exp(-dist ** 2 / (2. * (sigma * sigma)))

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
    # numba.cuda.synchronize()
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


# def adjacency_matrix_numba_wrapper(x_left, x_right, idx, one_matrix_rows, sigma):
#     W_shape = [1 + i for i in np.max(idx)]
#     W = np.zeros(W_shape, dtype='float64')
#     W_flat = W.ravel()
#     adjacency_matrix_numba(W_flat, x_left, x_right, len(idx), sigma, one_matrix_rows)
#     W = W_flat.reshape(W_shape)
#     return W


@numba.vectorize('float64(float64)')
def lat2y(lat):
    return math.log(math.tan(math.radians(lat) / 2 + math.pi/4)) * 6378137.0

@numba.vectorize('float64(float64)')
def lon2x(lon):
    return math.radians(lon) * 6378137.0

# Ary_in: first dimension is fid, second is rows, third is data
# ary out: first dimension fid, second rows, third data (but then scaled and averaged)
@numba.jit#(parallel=True)
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

# @numba.jit
def scale_and_average_df_numba_wrapper(df, sample_to_n_rows, fields=['lat', 'lon'], dtype='float32'):
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
    number_of_discared_dfs = gdf_as_numpy_arrays.shape[0] - converted_df.shape[0]
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

    return ary_out, index_map, number_of_discared_dfs

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


def fiedler_vector(L):
    l, U = eigh(L)
    f = U[:, 1]
    return f


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


def parallelize_df(df, func, n_partitions=10):
    df_split = np.array_split(df, n_partitions)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def parallelize_dict(d, func, *args, **kwargs):
    manager = multiprocessing.Manager()
    out_dict = manager.dict()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = [pool.apply_async(func, args=(out_dict, k, v, *args), kwds=kwargs) for k, v in d.items()]
        _ = [r.get() for r in results]
    return out_dict


def convert_df_to_proper_encoded_gdf(df):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    gdf = prepare_gdf_for_plotting(gdf)
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    return gdf


# def adjacency_matrix_multithread(args):
#     sigma = 26561*.3
#     return args[0], adjacency_matrix(*args[1], sigma)


def inverse_map(map):
    inv_map = {}
    for k, v in map.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map


if __name__ == "__main__":
    fid_to_cluster_map = None
    # fid_to_cluster_map = {'46e3c42e': 0, '4788ba10': 0, '47bffa98': 0, '487e09de': 0, '48ff044e': 0, '4982b898': 0, '49e4894c': 0, '4a774af2': 0, '4afc9e14': 0, '4b4eb14a': 0, '4c1bd5a8': 0, '4c6a5624': 0, '4d096700': 0, '4d5a048a': 0, '4da1561e': 0, '4f165918': 0, '4fcb90f8': 0, '4fd0426a': 0, '51448160': 0, '5180722e': 0, '51c0d36e': 0, '5253f40a': 0, '530208f6': 0, '54252c90': 0, '56689ff0': 0, '574cd0a8': 0, '5772b642': 0, '57a51204': 0, '580fb5c8': 0, '59040948': 0, '593eaf58': 0, '59f6388a': 0, '5bc975f0': 0, '5c0dc9a8': 0, '5c509a6c': 0, '5c67048c': 0, '5cce4958': 0, '5d814986': 0, '5e0cc6d2': 0, '5e12611e': 0, '5ede2886': 0, '5edf2e70': 0, '5f4c12b0': 0, '5f5fa806': 0, '5f8ed942': 0, '5fa591d6': 0, '5fd64ccc': 0, '602e64bc': 0, '604eaabe': 0, '61c2d078': 0, '61d19342': 0, '62caa158': 0, '632195d0': 0, '6342921c': 0, '638cfdd4': 0, '63b33364': 0, '63fac2f6': 0, '6447b6b0': 0, '64ede0a8': 0, '64fab12a': 0, '6546d280': 0, '65bb1690': 0, '66397260': 0, '664534ce': 0, '66669aec': 0, '67382850': 0, '6791240a': 0, '67c7bf2e': 0, '67d922e6': 0, '68360218': 0, '6896463c': 0, '68c2aee8': 0, '68ea810c': 0, '69018186': 0, '6908b582': 0, '69378f4c': 0, '6951d14a': 0, '69563a46': 0, '697c0fc8': 0, '69ace21a': 0, '69f3b136': 0, '6a469770': 0, '6aade1be': 0, '6b4f5c1a': 0, '6b800b9e': 0, '6bcfff28': 0, '6bd58b50': 0, '6bef6e58': 0, '6c3c4016': 0, '6c404634': 0, '6c9af65c': 0, '6c9dc76e': 0, '6cfa5ee8': 0, '6cfbbd88': 0, '6d0e4318': 0, '6d93e900': 0, '6dd450ee': 0, '6dec3498': 0, '6e1135c2': 0, '6e686ea0': 0, '6e7e3654': 0, '6ed2ef3c': 0, '6f224df2': 0, '6f365478': 0, '6f566de4': 0, '6f972294': 0, '6fbd4434': 0, '6fbd4492': 0, '6fcc4ee2': 0, '6ff93948': 0, '7013f968': 0, '704cc40a': 0, '70550c0a': 0, '70973fd0': 0, '70ee80b0': 0, '713335d4': 0, '7133de3a': 0, '719562d6': 0, '71d203b2': 0, '7256ef82': 0, '72e001e6': 0, '7330c590': 0, '73bc1a5a': 0, '74013a68': 0, '749d87ce': 0, '74c3db7c': 0, '7536f09e': 0, '75854b04': 0, '75a4c812': 0, '76085486': 0, '77503822': 0, '7835cf4a': 0, '7889891e': 0, '78d35b7a': 0, '792d4248': 0, '7a071fcc': 0, '7a37a8d2': 0, '7a481cac': 0, '7b1f596a': 0, '7bc3895e': 0, '7c08b092': 0, '7d1e20a2': 0, '7f06267c': 0, '7fa86512': 0, '8052a694': 0, '80b6a4aa': 0, '80d74d46': 0, '81018d1c': 0, '81597734': 0, '82075404': 0, '82416ce2': 0, '82f7f250': 0, '834c49a0': 0, '835148e6': 0, '880de9ca': 0, '88772b10': 0, '88890da8': 0, '88d6742c': 0, '88f306fe': 0, '8ad1f878': 0, '8b3fe73e': 0, '8c8db5e4': 0, '8f9eb076': 0, '8fedf8f2': 0, '908f28f8': 0, '90dbcca8': 0, '917885c0': 0, '9318656c': 0, '93497620': 0, '93bbfc22': 0, '956abb76': 0, '958b294c': 0, '95f127e2': 0, '96067b24': 0, '96302adc': 0, '963a067e': 0, '973b84e4': 0, '975bec3e': 0, '97c627a2': 0, '97d32a9c': 0, '97fd1410': 0, '9830b2ca': 0, '98437734': 0, '9893ec28': 0, '98e137da': 0, '98f8e6fa': 0, '99568904': 0, '99b79c3a': 0, '9a015b7c': 0, '9a019e84': 0, '9a1f7e40': 0, '9a25cf3e': 0, '9a39fa90': 0, '9a504fca': 0, '9ab6a9e6': 0, '9af55f2e': 0, '9b096a64': 0, '9b228d0a': 0, '9bd12f5e': 0, '9c6f3758': 0, '9c80bac8': 0, '9ccd7eda': 0, '9d3d4526': 0, '9dbdf784': 0, '9de796a2': 0, '9e1e3018': 0, '9e81a350': 0, '9ed392aa': 0, '9f127c54': 0, '9f21c876': 0, '9f2dcaae': 0, '9f8e82ea': 0, '9fcd28b0': 0, '9fdf238a': 0, '9fec6108': 0, 'a010b9fe': 0, 'a014ce5e': 0, 'a029f9d2': 0, 'a219ad46': 0, 'a277055e': 0, 'a291d104': 0, 'a2d2c5c4': 0, 'a31aada8': 0, 'a36ae782': 0, 'a3f71aea': 0, 'a400133e': 0, 'a446e962': 0, 'a511d46a': 0, 'a646aa18': 0, 'a6c1f0ce': 0, 'a6dce384': 0, 'a834d246': 0, 'a84fc858': 0, 'a86f6348': 0, 'a887f17e': 0, 'a8ccb5a2': 0, 'a903ddfc': 0, 'a958292a': 0, 'a9756940': 0, 'aaee3aa4': 0, 'ac17fc62': 0, 'ac8919ec': 0, 'ad327528': 0, 'adc05898': 0, 'add45b5e': 0, 'ade0e28e': 0, 'ae366768': 0, 'aeb03642': 0, 'aeba9a4c': 0, 'af680556': 0, 'afb5320e': 0, 'b09041f0': 0, 'b0cc9c2c': 0, 'b11fd72a': 0, 'b1318128': 0, 'b1c1bde2': 0, 'b23dad12': 0, 'b26a0fc4': 0, 'b26e2f6e': 0, 'b2939286': 0, 'b2be4404': 0, 'b32a5aae': 0, 'b3bfa47e': 0, 'b3f173b4': 0, 'b445a95c': 0, 'b46b7858': 0, 'b4dc3e1c': 0, 'b4fd462a': 0, 'b50fab08': 0, 'b6208846': 0, 'b708a892': 0, 'b79bbc04': 0, 'b834da38': 0, 'b83a5d14': 0, 'b8468cec': 0, 'b855a542': 0, 'b89a67a4': 0, 'b9445e58': 0, 'b998a53a': 0, 'b9c1b3bc': 0, 'ba275e4c': 0, 'bade3c70': 0, 'bb6ee702': 0, 'bb7b56f4': 0, 'bc684e32': 0, 'bc6b0578': 0, 'bca31346': 0, 'bcf99338': 0, 'bd3a7038': 0, 'bd96090c': 0, 'bdedc52a': 0, 'be2f1f84': 0, 'beadfe08': 0, 'bf39ac32': 0, 'bfa8a006': 0, 'bfd50a4c': 0, 'c04478fa': 0, 'c0558f5a': 0, 'c0aef928': 0, 'c15f836a': 0, 'c1de0866': 0, 'c2278270': 0, 'c2429254': 0, 'c29dc26e': 0, 'c2b62228': 0, 'c34a1d70': 0, 'c35b3a10': 0, 'c3aa6572': 0, 'c3beb892': 0, 'c3d75528': 0, 'c40865e6': 0, 'c41c22c0': 0, 'c4a308c6': 0, 'c4c5e09e': 0, 'c5002a42': 0, 'c53ee57a': 0, 'c57e0d22': 0, 'c5f966ac': 0, 'c66a8648': 0, 'c75be81c': 0, 'c8806416': 0, 'c8895b52': 0, 'c8b6fc92': 0, 'c8c107a0': 0, 'c8fb2098': 0, 'c968d7a0': 0, 'c9750d40': 0, 'c9c90486': 0, 'c9ec88de': 0, 'ca77a7b6': 0, 'cb233d56': 0, 'cb41d2fc': 0, 'cb6b8d18': 0, 'cbcd41fc': 0, 'cc18e846': 0, 'cc4b484a': 0, 'cc5a90fc': 0, 'ccb134ca': 0, 'ccf125b2': 0, 'cd3a1380': 0, 'cd91da3e': 0, 'cdb9f0be': 0, 'cddac244': 0, 'ce9f1e64': 0, 'cefa2f2a': 0, 'cf7bf5f0': 0, 'cff96e4a': 0, 'd0009882': 0, 'd0228f46': 0, 'd06506f0': 0, 'd0b31dae': 0, 'd0cdf9a8': 0, 'd176c81c': 0, 'd18b7208': 0, 'd1e4bcc8': 0, 'd251dce0': 0, 'd2aa7f94': 0, 'd2fc251a': 0, 'd314e262': 0, 'd35755ac': 0, 'd35d504c': 0, 'd39e8558': 0, 'd40a6ab6': 0, 'd46897c6': 0, 'd541a962': 0, 'd57d38b0': 0, 'd58c8eb4': 0, 'd5c2281c': 0, 'd5d4ad66': 0, 'd651361a': 0, 'd688c8be': 0, 'd6ea3ee6': 0, 'd7713c98': 0, 'd7beacee': 0, 'd80e51b8': 0, 'd823e6ea': 0, 'd87ff160': 0, 'd969aada': 0, 'd9d6f0d6': 0, 'd9ecd7b6': 0, 'da5b74aa': 0, 'da9bd9f0': 0, 'daba1d3e': 0, 'dad13faa': 0, 'db354d42': 0, 'db85e70c': 0, 'dbee5b5c': 0, 'dc385dce': 0, 'dc81d8dc': 0, 'dc9f5006': 0, 'dd29f8be': 0, 'dd7e9f0e': 0, 'dd8fdf76': 0, 'ddce9806': 0, 'ddd973f2': 0, 'de560f5c': 0, 'de5caf56': 0, 'de75f196': 0, 'de7c3ae2': 0, 'df021112': 0, 'df66c738': 0, 'df982fc6': 0, 'dfaff64c': 0, 'dfd400fa': 0, 'dff29164': 0, 'dff4bc96': 0, 'e0057658': 0, 'e034b1f2': 0, 'e041a574': 0, 'e0afd0ee': 0, 'e20ca0b6': 0, 'e28043ea': 0, 'e32ca856': 0, 'e3483a9e': 0, 'e40ec5b0': 0, 'e47c4630': 0, 'e4e25380': 0, 'e4ef08dc': 0, 'e5423606': 0, 'e549a2ce': 0, 'e54ed762': 0, 'e54f9594': 0, 'e5830ef6': 0, 'e64d4ba8': 0, 'e76b6ec0': 0, 'e7b915da': 0, 'e7f88422': 0, 'e86c79e0': 0, 'e877b35a': 0, 'e8c17b7a': 0, 'e9546d36': 0, 'e9e4801a': 0, 'e9f16c6c': 0, 'ea6eda6c': 0, 'ea94f5b2': 0, 'eac9236e': 0, 'eaf5051a': 0, 'ebf54c18': 0, 'ebfe8e2c': 0, 'eca57b4c': 0, 'ecbf3672': 0, 'ed10b826': 0, 'ed34ae2a': 0, 'ed5d2e9a': 0, 'ed94401a': 0, 'edf08b04': 0, 'ee6b891c': 0, 'eeaac4ec': 0, 'eeafcf1e': 0, 'eed08934': 0, 'ef3399a2': 0, 'f0383b50': 0, 'f040c3c4': 0, 'f0a9cfae': 0, 'f0b8bf14': 0, 'f15bbaf2': 0, 'f1f66c00': 0, 'f21014a2': 0, 'f24ed642': 0, 'f299c332': 0, 'f2d7cefc': 0, 'f2ddb574': 0, 'f356732e': 0, 'f372f2c4': 0, 'f3a3c520': 0, 'f3b4b7c2': 0, 'f41a8bce': 0, 'f499e2ca': 0, 'f54b2ecc': 0, 'f57e2480': 0, 'f5ef8396': 0, 'f69393dc': 0, 'f6a05ffe': 0, 'f7202c5c': 0, 'f746c7c2': 0, 'f75cbdb6': 0, 'f783f1c4': 0, 'f803d984': 0, 'f85183d2': 0, 'f8a6bc58': 0, 'f8de434e': 0, 'f93af062': 0, 'f93bf124': 0, 'f96a7fee': 0, 'f9d11330': 0, 'f9dc76bc': 0, 'f9de43c0': 0, 'f9e05430': 0, 'fa38c8cc': 0, 'fa4c1ff8': 0, 'fa8965de': 0, 'fae379fc': 0, 'fb822020': 0, 'fbac7366': 0, 'fc15ccbc': 0, 'fc38d612': 0, 'fc7568fc': 0, 'fc7897ac': 0, 'fcd1daec': 0, 'fcdb51c6': 0, 'fcf3ed58': 0, 'fd805d1a': 0, 'fdf449e6': 0, 'fdfb3b16': 0, 'fe41db02': 0, 'fe5d704c': 0, 'fe88e7e0': 0, 'fef44da0': 0, 'fef789ca': 0, 'fef9e576': 0, 'ff1c9a3a': 0, 'ffa96500': 0, 'ffeac662': 0, 'fffffc26': 0, '428740ea': 1, '4311aeb0': 1, '43d6033c': 1, '44709122': 1, '44b42036': 1, '4509051a': 1, '45f1a73e': 1, '45f6868c': 1, '46538a58': 1, '467fefa8': 1, '47180c66': 1, '4783c3f2': 1, '479f2e44': 1, '47a9965e': 1, '490a7b62': 1, '499526ae': 1, '4995a232': 1, '4c569b3e': 1, '4cdb106c': 1, '4d909360': 1, '4e6f0550': 1, '4edb6556': 1, '4f855eda': 1, '5018fc62': 1, '515f5ba2': 1, '51c2160c': 1, '52272fa6': 1, '52874422': 1, '53758e66': 1, '5407a170': 1, '54379e48': 1, '543afc46': 1, '5444c852': 1, '565214a6': 1, '575e40b8': 1, '5797e07a': 1, '58ee148a': 1, '595e1cbc': 1, '5a8e5fa2': 1, '5abeef0a': 1, '5b772732': 1, '5b8f7e68': 1, '5cdf5658': 1, '5e818684': 1, '5ec6a764': 1, '5f8a390e': 1, '5fd9ce10': 1, '62166c7e': 1, '649a1874': 1, '661bc1e8': 1, '684b1a9a': 1, '6b61bc3e': 1, '6ce6eb24': 1, '6fd6da6a': 1, '70113e76': 1, '738b5650': 1, '74743988': 1, '74fbdafa': 1, '76d2335a': 1, '77912888': 1, '77d4fda0': 1, '7802c80c': 1, '78b381c4': 1, '78c2c4a4': 1, '78d4acce': 1, '78e00da2': 1, '7952a330': 1, '7a1b333c': 1, '7a20c7e2': 1, '7a9ec6d4': 1, '7b576d96': 1, '7b74b1f4': 1, '7c8fdd16': 1, '7cb7af9e': 1, '7cdb4b34': 1, '7d044ccc': 1, '7dafe2b8': 1, '7dc98628': 1, '7f025e52': 1, '7fc6a53c': 1, '81007860': 1, '810a9d36': 1, '810ed7ca': 1, '81821730': 1, '8197afea': 1, '837c9c4a': 1, '8527b656': 1, '85cd62ea': 1, '867871ee': 1, '86ec1e64': 1, '88461a3a': 1, '8b22d504': 1, '8b2b6354': 1, '8b9220c0': 1, '8b9515ba': 1, '8be8c1ec': 1, '8c0cd1d6': 1, '8c8e2b14': 1, '8da607a6': 1, '8e0663a8': 1, '8e2e04ee': 1, '8fba8576': 1, '905384a6': 1, '91a7f31e': 1, '91b5802e': 1, '92967214': 1, '93558f96': 1, '935e9460': 1, '949fbda4': 1, '94e13752': 1, '96603aa6': 1, '992a6716': 1, '99396310': 1, '9b5a9ad8': 1, '9bb43b56': 1, '9d5d07f8': 1, '9da10f7a': 1, '9dc92a1e': 1, '9de22244': 1, '9ebe63ee': 1, 'a221def8': 1, '213d7b5c': 2, '21f62940': 2, '223a168c': 2, '22b6b76e': 2, '2331265c': 2, '238995d0': 2, '23f51a9e': 2, '244376ee': 2, '24a65d86': 2, '24f52768': 2, '25485a6e': 2, '2592e0f2': 2, '263e9906': 2, '268b4a30': 2, '26bacdf0': 2, '26ecf0b4': 2, '276e915a': 2, '27e07144': 2, '28697a70': 2, '28b39bf0': 2, '29171a36': 2, '299e98d0': 2, '29b8181e': 2, '2aaed2c6': 2, '2b2757f0': 2, '2bc3c6a8': 2, '2bc4bd6a': 2, '2c52fae4': 2, '2ca28ff0': 2, '2d16823e': 2, '2d641eea': 2, '2e1169ce': 2, '2eae9c62': 2, '2ffa3efa': 2, '3022f688': 2, '3065b4fa': 2, '30d01a5c': 2, '311a9320': 2, '317a4284': 2, '31e8c664': 2, '32765038': 2, '32d233f8': 2, '332a0d62': 2, '336b57fe': 2, '33e0ded4': 2, '344b6a9c': 2, '34de0992': 2, '352f7b4c': 2, '35928408': 2, '35c2d59a': 2, '36120be2': 2, '362a9cac': 2, '3656a108': 2, '369f453e': 2, '36c345ec': 2, '36f0d246': 2, '374767f0': 2, '37af64a4': 2, '3848bb90': 2, '3898df76': 2, '393a7d36': 2, '394feafe': 2, '39870d40': 2, '3a36078c': 2, '3aca3ad8': 2, '3b4e6db2': 2, '3b5d5a66': 2, '3b9caf18': 2, '3c373402': 2, '3cf57f84': 2, '3d0d0ed8': 2, '3d4ea32a': 2, '3d96934c': 2, '3e0bc252': 2, '3e853b64': 2, '3eaa6c7c': 2, '3ec4f2d6': 2, '3f5f7c98': 2, '3f6658c4': 2, '3fcf5b30': 2, '403c6658': 2, '40ce9a78': 2, '40e442a6': 2, '41b927e6': 2, '41d2d8b2': 2, '428b6dbe': 2, '42a7aba0': 2, '42e5581a': 2, '443a6b4c': 2, '44433600': 2, '454ed400': 2, '45574dce': 2, '45984aa4': 2, '4619070c': 2, '46349120': 2, '463ee6c0': 2, '1f848030': 3, '1fd4cd2e': 3, '201e8f68': 3, '205272ba': 3, '20ac0d3e': 3, '20db7880': 3, '20dfc7be': 3, '20f5bfd8': 3, '20fa38a6': 3, '24fb3090': 3, '25f7cdb4': 3, '26885cd0': 3, '279ae0e8': 3, '2c37f0d2': 3, '2eba663c': 3, '2f813b5e': 3, '309b746e': 3, '356fb9dc': 3, '35e83b64': 3, '35fc46cc': 3, '361ec97c': 3, '36979dc0': 3, '37917584': 3, '39b001aa': 3, '3a584ea0': 3, '3ab3c1e0': 3, '3b037c08': 3, '3b0979fa': 3, '3c63acd0': 3, '3db49dc4': 3, '3e213466': 3, '4021a07a': 3, '1efce076': 4, '1f0adb04': 4, '1f18473a': 4, '12fb7ea4': 5, '13e69010': 5, '15c3f2ba': 5, '15cb0e9c': 5, '15e74ab2': 5, '166f563c': 5, '168c4292': 5, '16f997ac': 5, '17b12886': 5, '18209ef0': 5, '1847e8ac': 5, '1856df56': 5, '18a69c4e': 5, '18d579ec': 5, '197612c6': 5, '199eac22': 5, '19e307a0': 5, '19e7eb62': 5, '19facfd4': 5, '1a234748': 5, '1a7a9020': 5, '1b01ff10': 5, '1b61f69a': 5, '1bca4268': 5, '1d394536': 5, '1d88988e': 5, '1deee364': 5, '1e662f6e': 5, '1f164cd2': 5, '1f1d23ea': 5, '117cce2a': 6, '124ae10c': 7, '128ad6f4': 7, '12fcfba8': 7, '138e3226': 7, '13a75ac6': 7, '13caf422': 7, '14866766': 7, '1591d690': 7, '15e67790': 7, '173e767e': 7, '18e14a60': 7, '18f8264a': 7, '1a198ca8': 7, '11ce725c': 8, '066e6c50': 9, '07043780': 9, '07bf365c': 9, '07fd4f5a': 9, '08473520': 9, '08a7a5d6': 9, '08fb62d4': 9, '0981b12c': 9, '09c5ca38': 9, '09e85698': 9, '0a0e5d48': 9, '0a235982': 9, '0a768e72': 9, '0ad86f2a': 9, '0bce80d6': 9, '0d636524': 9, '0db1a28e': 9, '0e982a42': 9, '0ef61a08': 9, '0f64c868': 9, '0fe6b8fa': 9, '102afdee': 9, '103f67f2': 9, '106116fe': 9, '0127cdd6': 10, '003a4a52': 11, '0165683a': 12, '027a95b0': 12, '023ab882': 13, '02c32032': 14, '02d0d7ea': 15, '02efe3a6': 15, '03081c6e': 15, '03dab69c': 15, '04a062b6': 15, '056b4800': 15, '0321283a': 16, '03b7b11a': 16, '041e7c56': 16, '04abe4b0': 16, '0531d336': 16, '0578cfb6': 16, '05ba9c48': 16, '05fca62e': 17, '05e4d562': 18, '071a32e2': 19, '06acb0c8': 20, '08282ce8': 21, '0a73b58a': 21, '08335bfe': 22, '0ae1874a': 23, '0b8d1ede': 24, '0da31f3e': 24, '0f9ecc70': 24}

    airspace_query = "airport=='EHAM'"
    zoom = 15

    # fields = ('x', 'y', 'alt')
    fields = ['lat', 'lon']
    airspace = ehaa_airspace.query(airspace_query)
    n_data_points = 100
    one_matrix_shape = n_data_points, len(fields)
    # sigma = 1
    minalt = 200  # ft
    maxalt = 10000

    sigma = 4000.
    omega_min = 1

    # stop = lambda X, Y: np.max(X) / np.max(Y) < omega_min
    stop = lambda X, Y: np.var(X)/np.var(Y) < omega_min
    log("Start reading csv")

    # #### SINGLE FILENAME
    df = pd.read_csv('data/adsb_decoded_in_eham/ADSB_DECODED_20180101.csv.gz')  # , converters={'callsign': lambda s: s.replace('_', '')})

    # #### ENTIRE DIRECTORY
    # path = './data/adsb_decoded_in_eham/'
    # max_number_of_files = 15
    # df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))[:max_number_of_files]))
    log("Completed reading CSV")

    orig_df = copy.deepcopy(df)

    if fid_to_cluster_map is None:

        if minalt is not None:
            df = df[df['alt'] > minalt]
        if maxalt is not None:
            df = df[df['alt'] < maxalt]
        # fids = ['07bf365c', '0db1a28e', '19e307a0', '1deee364', '20f5bfd8', '31e8c664', '40e442a6', '41b927e6', '5a8e5fa2', '704cc40a', '9de796a2', '9e1e3018', 'c2429254', 'c9c90486', 'df982fc6']

        gdf = parallelize_df(df, convert_df_to_proper_encoded_gdf)
        # gdf = gdf.query('fid in @fids')
        log("Converted to GDF with proper encoding")
        df_g = tuple(gdf.groupby('fid'))

        data_points_per_group = np.array([len(v) for k, v in df_g])


        # num_groups = len(df_g)
        log("Completed groupby")
        # fid_list = np.empty(num_groups, dtype='U10')
        x_numba, fid_list_numba, number_of_discared_dfs = scale_and_average_df_numba_wrapper(df, n_data_points, fields,
                                                                                       dtype='float64')
        processed_dfs = parallelize_dict({fid_name: fid_df for fid_name, fid_df in df_g}, scale_and_average_df_multi, n_data_points=n_data_points, fields=fields)
        fields = ['x', 'y']

        log("Thew away {0} fid's".format(number_of_discared_dfs))
        fid_list = np.array(processed_dfs.keys())

        # processed_df_i = 0
        # for fid_name, fid_df in df_g:
        #     fid_df_processed = scale_and_average_df(fid_df)
        #     if fid_df_processed is not None:
        #         processed_dfs[fid_name] = fid_df_processed
        #         fid_list[processed_df_i] = fid_name
        #         processed_df_i += 1

        # fid_list = fid_list[:processed_df_i]
        num_groups = len(fid_list)

        log("Processed groups")
        i = 0
        # adjacency_matrices_to_calculate = []
        x_l = []
        n = len(processed_dfs)
        one_matrix_rows, one_matrix_cols = one_matrix_shape
        processed_dfs_list = list(processed_dfs.values())
        x_shape = tuple((len(processed_dfs_list), *one_matrix_shape))
        x = np.zeros(x_shape, dtype='float64')
        for i in range(x.shape[0]):
            x[i, :, :] = processed_dfs_list[i]

        log("Queued adjacency matrix calculation")
        #
        # W = np.zeros((n, n), dtype='float64')
        # # Function below modifies W in place
        # adjacency_matrix_numba(W, x, sigma)
        # log("Calculated adjacency matrix")
        # result = []
        #
        # spectralCluster(W, stop, result)
        # fid_to_cluster_map = {}
        # for cluster_number, fidlist_indices in enumerate(result):
        #     for fid in fid_list[fidlist_indices.ravel()]:
        #         fid_to_cluster_map[fid] = cluster_number
        #
        # gdf['cluster'] = gdf['fid'].map(fid_to_cluster_map)
        # log("Determined fid_to_cluster_map")
        # log("{0} clusters found (sigma={1}, omega_min={2})".format(1+max(fid_to_cluster_map.values()), 1, omega_min))
        #
        # cluster_to_fid_map = inverse_map(fid_to_cluster_map)
        # df_group_dict = {k: v for k, v in df_g}
        # # Sort from largest to smallest cluster
        # for key, fids in sorted(cluster_to_fid_map.items(), key=lambda a: len(a[1]))[::-1]:
        #     tracks_concat_shape = len(fids), processed_dfs[fids[0]].shape[0], processed_dfs[fids[0]].shape[1]
        #     tracks_concat_dtype = processed_dfs[fids[0]].dtype
        #     tracks_concat = np.empty(tracks_concat_shape, dtype=tracks_concat_dtype)
        #     for fid_i in range(tracks_concat_shape[0]):
        #         tracks_concat[fid_i, :, :] = processed_dfs[fids[fid_i]]
        #     tracks_concat_flat = tracks_concat.reshape((-1, tracks_concat_shape[2]))
        #     tracks_mean = tracks_concat.mean(axis=0)
        #
        #     fig = plt.figure()
        #     airspace_projected = prepare_gdf_for_plotting(airspace)
        #     ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
        #     # add_basemap(ax, zoom=zoom, ll=False)
        #     ax.set_axis_off()
        #     colorcycle = cycle(['C0', 'C1'])
        #     sizecycle = cycle([1, 0.1])
        #     for tracks in [tracks_mean, tracks_concat_flat]:
        #         #gdf_for_plotting = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))
        #         #gdf_track_converted = gdf_for_plotting  # prepare_gdf_for_plotting(gdf)
        #         color = next(colorcycle)
        #         size = next(sizecycle)
        #         #gdf_track_converted.plot(ax=ax, color=color, markersize=size, linewidth=size)
        #         gs = geopandas.GeoSeries(geopandas.points_from_xy(tracks[:, 0], tracks[:, 1]))
        #         gs.crs = {'init': 'epsg:3857', 'no_defs': True}
        #         gs.plot(ax=ax, color=color, markersize=size, linewidth=size)
        #     plt.show()
        #     if input("Continue? [y/n]").capitalize() == "N":
        #         break
        # # if input("Save vars to pickle? [y/n]").capitalize() == "Y":
        # #     out_fn = "clustering_vars.pkl"
        # #     log("Saving to {0}".format(out_fn))
        # #     with open(out_fn, 'wb') as f:
        # #         import pickle
        # #         pickle.dump(df, f)
        # #         pickle.dump(gdf, f)
        # #         pickle.dump(processed_dfs, f)
        # #         pickle.dump(fid_list, f)
        # #         pickle.dump(x, f)
        # #         pickle.dump(lower_indices_list, f)
        # #         pickle.dump(W, f)
        # #         pickle.dump(fid_to_cluster_map, f)
        # #         pickle.dump(cluster_to_fid_map, f)
        #

    # # ## VISUALISATION
    # #
    # # df = orig_df
    # # df['cluster'] = df['fid'].map(fid_to_cluster_map)
    # # if minalt is not None:
    # #     df = df[df['alt'] > minalt]
    # # if maxalt is not None:
    # #     df = df[df['alt'] < maxalt]
    # #
    # # df_alt_min, df_alt_max = df['alt'].min(), df['alt'].max()
    # # fig = plt.figure()
    # # airspace_projected = prepare_gdf_for_plotting(airspace)
    # # ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    # # # add_basemap(ax, zoom=zoom, ll=False)
    # # ax.set_axis_off()
    # #
    # # gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    # # gdf_track_converted = prepare_gdf_for_plotting(gdf)
    # # # gdf_track_converted.plot(ax=ax, column='alt', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1, vmin=df_alt_min, vmax=df_alt_max)
    # # gdf_track_converted.dissolve('cluster').plot(ax=ax, label='cluster', markersize=0.1, linewidth=0.1, column='cluster')
    # # # plt.show()
    # # handles, labels = ax.get_legend_handles_labels()  # return clusters and labels
    # # interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
    # #                                                          ax.collections),
    # #                                                      labels,
    # #                                                      # alpha_unsel=0.5,
    # #                                                      # alpha_over=1.5,
    # #                                                      start_visible=True)
    # # plugins.connect(fig, interactive_legend)
    # # mpld3.show()
    # #
    #
