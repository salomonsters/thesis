import numpy as np
import scipy as sp
from scipy.linalg import eigh
import pandas as pd
import itertools
import scipy as sp
import geopandas
from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import datetime
import glob
import os
import multiprocessing

def log(m):
    print("{time}: {0}".format(m, time=datetime.datetime.now()))

# TODO: then maybe pad as well so we don't get nan's?
# TODO: perhaps normalize to a track with a specific airspeed so we have more comparable tracks
# and then don't group on length?

def adjacency_matrix(x_i, x_j, sigma):
    x = x_i - x_j
    x.dropna(inplace=True)
    if len(x) == 0:
        return 0
    dist = np.linalg.norm(x)
    return np.exp(-dist ** 2 / (2. * (sigma * sigma)))


def scale_and_average_df(df, fields=('lon', 'lat')):
    if len(df) == 1:
        # We cannot scale and average in this case, so return None
        return None
    ts_min, ts_max = df['ts'].min(), df['ts'].max()
    ts_len = ts_max - ts_min
    ts_scaled = (df['ts'] - ts_min)/ts_len*n_data_points
    df.loc[:,'timestamp'] = ts_scaled.apply(lambda x: pd.Timestamp(x, unit='s'))
    df.set_index('timestamp', inplace=True)
    resampled = df.resample(rule='10s')[fields].mean()
    return resampled


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

    if not stop_function(W_il_il) and not np.array_equal(original_indices[i_l], original_indices):
        spectralCluster(W_il_il, stop_function, result_indices, original_indices[i_l])
    else:
        result_indices.append(original_indices[i_l])
    if not stop_function(W_ir_ir) and not np.array_equal(original_indices[i_r], original_indices):
        spectralCluster(W_ir_ir, stop_function, result_indices, original_indices[i_r])
    else:
        result_indices.append(original_indices[i_r])


def parallelize_df(df, func, n_partitions=10):
    df_split = np.array_split(df, n_partitions)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def convert_df_to_proper_encoded_gdf(df):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    return gdf


def adjacency_matrix_multithread(args):
    return args[0], adjacency_matrix(*args[1], sigma)

if __name__ == "__main__":
    airspace_query = "airport=='EHAM'"
    # zoom = 15
    #
    airspace = ehaa_airspace.query(airspace_query)
    n_data_points = 1000
    sigma = 1
    minalt = 1000  # ft
    maxalt = 5000

    # #### SINGLE FILENAME
    df = pd.read_csv('data/adsb_decoded/ADSB_DECODED_20180101.csv.gz')#, converters={'callsign': lambda s: s.replace('_', '')})
    log("Read CSV")
    # #### ENTIRE DIRECTORY
    # path = './data/adsb_decoded/'
    # max_number_of_files = 5
    # df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))[:max_number_of_files]))
    import copy
    orig_df = copy.deepcopy(df)

    if minalt is not None:
        df = df[df['alt'] > minalt]
    if maxalt is not None:
        df = df[df['alt'] < maxalt]

    gdf = parallelize_df(df, convert_df_to_proper_encoded_gdf)

    log("Converted to GDF with proper encoding")
    df_g = tuple(gdf.groupby('fid'))
    num_groups = len(df_g)

    fid_list = np.empty(num_groups, dtype='U10')

    processed_dfs = dict()
    processed_df_i = 0
    for fid_name, fid_df in df_g:
        fid_df_processed = scale_and_average_df(fid_df, fields=['x', 'y'])
        if fid_df_processed is not None:
            processed_dfs[fid_name] = fid_df_processed
            fid_list[processed_df_i] = fid_name
            processed_df_i += 1

    fid_list = fid_list[:processed_df_i]
    num_groups = processed_df_i

    W = np.zeros((num_groups, num_groups), dtype=float)
    log("Processed groups")
    i = 0
    adjacency_matrices_to_calculate = []
    for fid_i, x_i in processed_dfs.items():
        j = 0
        for fid_j, x_j in processed_dfs.items():
            # TODO perhaps only have lower triangular indices here? And then mirror?
            adjacency_matrices_to_calculate.append(((i, j), (x_i, x_j)))
            j += 1
        i += 1
    log("Queued adjacency matrix calculation")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    calculated_adjacency_matrices = pool.map(adjacency_matrix_multithread, adjacency_matrices_to_calculate)
    pool.close()
    pool.join()

    for index_tuple, result in calculated_adjacency_matrices:
        W[np.ix_(index_tuple)] = result
    log("Calculated adjacency matrices")
    result = []
    omega_min = 0.1
    stop = lambda X: not (np.var(X) > omega_min)
    spectralCluster(W, stop, result)
    fid_to_cluster_map = {}
    for cluster_number, fidlist_indices in enumerate(result):
        for fid in fid_list[fidlist_indices.ravel()]:
            fid_to_cluster_map[fid] = cluster_number

    gdf['cluster'] = gdf['fid'].map(fid_to_cluster_map)
    log("Determined fid_to_cluster_map")
    ## VISUALISATION

    df = orig_df
    df['cluster'] = df['fid'].map(fid_to_cluster_map)
    if minalt is not None:
        df = df[df['alt'] > minalt]
    if maxalt is not None:
        df = df[df['alt'] < maxalt]

    df_alt_min, df_alt_max = df['alt'].min(), df['alt'].max()
    fig = plt.figure()
    airspace_projected = prepare_gdf_for_plotting(airspace)
    ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    # add_basemap(ax, zoom=zoom, ll=False)
    ax.set_axis_off()

    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    gdf_track_converted = prepare_gdf_for_plotting(gdf)
    # gdf_track_converted.plot(ax=ax, column='alt', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1, vmin=df_alt_min, vmax=df_alt_max)
    gdf_track_converted.plot(ax=ax, column='cluster', cmap='plasma', legend=False, markersize=0.1, linewidth=0.1)
    plt.show()
