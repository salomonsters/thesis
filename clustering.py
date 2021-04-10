import copy
from itertools import cycle
import warnings

import geopandas
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from pyproj import CRS

from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting
from tools import create_logger

from numba import float64, int_

Adjacency_spec = [
    ('W', float64[:, :]),
    ('D', float64[:, :]),
    ('x', float64[:, :, :]),
    ('K', int_),
    ('sigmas', float64[:]),
]


@numba.jitclass(Adjacency_spec)
class Adjacency:
    def __init__(self, x, K):
        self.W = np.zeros((x.shape[0], x.shape[0]))
        self.D = np.zeros((x.shape[0], x.shape[0]))
        self.sigmas = np.zeros(x.shape[0])
        self.K = K
        self.x = x

    def calculate_W(self):
        self.calculate_distance_matrix()
        for i in numba.prange(self.W.shape[0]):
            for j in numba.prange(self.W.shape[0]):
                if i == j:
                    # Value on diagonal is always 1
                    self.W[i, j] = 1.
                # We copy everything below the diagonal to above
                elif not (i < j):
                    dist_squared = self.D[i, j]
                    sigma_i_j_product = self.sigmas[i] * self.sigmas[j]
                    self.W[i, j] = math.exp(-dist_squared / (2. * sigma_i_j_product))
                    self.W[j, i] = self.W[i, j]

    def calculate_distance_matrix(self):
        for i in numba.prange(self.D.shape[0]):
            for j in numba.prange(self.D.shape[0]):
                if i == j:
                    # Value on diagonal is always 0
                    self.D[i, j] = 0.
                # We copy everything below the diagonal to above
                elif not (i < j):
                    dist_squared = 0.
                    for k in range(self.x.shape[1]):
                        for l in range(self.x.shape[2]):
                            dist_squared = dist_squared + (self.x[i, k, l] - self.x[j, k, l]) * (
                                    self.x[i, k, l] - self.x[j, k, l])
                    self.D[i, j] = dist_squared
                    self.D[j, i] = dist_squared
        for i in numba.prange(self.D.shape[0]):
            j = np.argsort(self.D[i, :])[self.K - 1]
            self.sigmas[i] = math.sqrt(self.D[i, j])


@numba.vectorize('float64(float64)')
def lat2y(lat):
    return math.log(math.tan(math.radians(lat) / 2 + math.pi / 4)) * 6378137.0


@numba.vectorize('float64(float64)')
def lon2x(lon):
    return math.radians(lon) * 6378137.0


@numba.njit
def array_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


@numba.njit(parallel=True)
def select_rows_and_columns(matrix, rows, columns):
    out = np.zeros((len(rows), len(columns)), dtype=matrix.dtype)
    for i in numba.prange(len(rows)):
        for j in numba.prange(len(columns)):
            out[i, j] = matrix[rows[i], columns[j]]
    return out


class Clustering:
    visualisation = None

    def __init__(self, x, K, e_mean, e_var, min_cluster_size=None, stop_function=None, visualisation=None):
        self.e_mean = e_mean
        self.e_var = e_var
        if min_cluster_size is None:
            min_cluster_size = 1
        self.min_cluster_size = min_cluster_size
        if visualisation:
            self.visualisation = visualisation
        if stop_function:
            self.stop_function = stop_function
        else:
            # self.stop_function = lambda X, Y: np.var(X)/np.var(Y) < omega_min
            # self.stop_function = lambda X, Y: np.max(X) / np.max(Y) < 4*omega_min

            # self.stop_function = lambda X, Y: np.std(X) < omega_min
            self.stop_function = lambda X, Y: np.var(X) / np.var(Y) < e_var or np.mean(X) > e_mean
        self.result_indices = None
        self.n_clusters = 0

        self.x = x
        self.K = K

    @staticmethod
    def fiedler_vector(W):
        D = np.zeros_like(W)
        for i in range(W.shape[0]):
            D[i, i] = np.sum(W[i, :])
        L = D - W
        l, U = np.linalg.eigh(L)

        v = U[:, 1]
        return v

    def calculate_adjacency(self):
        adjacency = Adjacency(self.x, self.K)
        adjacency.calculate_W()
        W = adjacency.W
        return W

    def spectral_cluster(self, W, original_indices=None, recursion=''):
        if self.result_indices is None:
            self.result_indices = np.zeros(W.shape[0], dtype='int')
        if original_indices is None:
            original_indices = np.array(range(W.shape[0]))
        v = self.fiedler_vector(W)

        # Indices of V with positive elements
        i_l = np.where(v >= 0)[0]
        i_r = np.where(v < 0)[0]
        W_il_il = select_rows_and_columns(W, i_l, i_l)
        W_ir_ir = select_rows_and_columns(W, i_r, i_r)

        if len(i_l) == 0 or len(i_r) == 0:
            warnings.warn(
                "Fiedler vector has same sign for all entries, no meaningful clustering possible for indices {}".format(
                    original_indices))
            return

        assert not array_equal(original_indices[i_l], original_indices)
        assert not array_equal(original_indices[i_r], original_indices)

        if self.visualisation:
            title = "green: n={0}, mean={1:.4f}, std={2:.4f}; ".format(len(i_l), np.mean(get_non_diagonal_elements(W_il_il)),
                                                                      np.std(get_non_diagonal_elements(W_il_il))) + \
                    "red: n={0}, mean={1:.4f}, std={2:.4f}".format(len(i_r), np.mean(get_non_diagonal_elements(W_ir_ir)),
                                                                      np.std(get_non_diagonal_elements(W_ir_ir))) + \
                    "total: n={0}, mean={1:.4f}, std={2:.4f}".format(len(i_l) + len(i_r), np.mean(get_non_diagonal_elements(W)),
                                                                   np.std(get_non_diagonal_elements(W)))
            self.visualisation.intermediate_result(self.x[original_indices[i_l]], self.x[original_indices[i_r]], title,
                                                   fname_arg=recursion)
        if self.stop_function(W_il_il, W) or len(i_l) < self.min_cluster_size:
            if len(i_l) < self.min_cluster_size:
                self.result_indices[original_indices[i_l]] = -1
            else:
                self.n_clusters += 1
                self.result_indices[original_indices[i_l]] = self.n_clusters
                if self.visualisation:
                    title = "n={0}, mean={1:.4f}, std={2:.4f}".format(len(i_l), np.mean(get_non_diagonal_elements(W_il_il)),
                                                                      np.std(get_non_diagonal_elements(W_il_il)))
                    self.visualisation.plot_cluster(self.x[original_indices[i_l]], title,
                                                    fname_arg="{0}_A_{1}".format(recursion, self.n_clusters), is_intermediate=True)
        else:
            self.spectral_cluster(W_il_il, original_indices[i_l], recursion + 'A')
        if self.stop_function(W_ir_ir, W) or len(i_r) < self.min_cluster_size:
            if len(i_r) < self.min_cluster_size:
                self.result_indices[original_indices[i_r]] = -1
            else:
                self.n_clusters += 1
                self.result_indices[original_indices[i_r]] = self.n_clusters
                if self.visualisation:
                    title = "n={0}, mean={1:.4f}, std={2:.4f}".format(len(i_r), np.mean(get_non_diagonal_elements(W_ir_ir)),
                                                                      np.std(get_non_diagonal_elements(W_ir_ir)))
                    self.visualisation.plot_cluster(self.x[original_indices[i_r]], title,
                                                    fname_arg="{0}_B_{1}".format(recursion, self.n_clusters), is_intermediate=True)
        else:
            self.spectral_cluster(W_ir_ir, original_indices[i_r], recursion + 'B')

    # Ary_in: first dimension is fid, second is rows, third is data
    # ary out: first dimension fid, second rows, third data (but then scaled and averaged)
    @staticmethod
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

    @classmethod
    def scale_and_average_df_numba_wrapper(cls, df, sample_to_n_rows, fields=('lat', 'lon'), dtype='float64',
                                           alt_conversion=1.):
        for field_counter, field in enumerate(fields):
            fields = list(fields)
            if field == 'lat':
                df['x'] = lon2x(df['lon'].values)
                fields[field_counter] = 'x'
            if field == 'lon':
                df['y'] = lat2y(df['lat'].values)
                fields[field_counter] = 'y'
            if field == 'alt':
                df['alt_scaled'] = df['alt'] * alt_conversion
                fields[field_counter] = 'alt_scaled'

        df_grouped = df.set_index('fid')[fields].groupby('fid')
        index_map_naive = np.array(df_grouped.count().index, dtype='str')
        gdf_as_numpy_arrays = df_grouped.apply(pd.DataFrame.to_numpy)
        rows_per_numpy_array = [_.shape[0] for _ in gdf_as_numpy_arrays]
        converted_df = np.array(
            [gdf_as_numpy_arrays[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows >= sample_to_n_rows])
        discarded_fids = np.array(
            [index_map_naive[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows < sample_to_n_rows])
        index_map = np.array(
            [index_map_naive[i] for i, n_rows in enumerate(rows_per_numpy_array) if n_rows >= sample_to_n_rows])
        max_n_datapoints = np.max(rows_per_numpy_array)
        n_fields = len(fields)
        ary_in_shape = converted_df.shape[0], max_n_datapoints, n_fields
        ary_in = np.zeros(ary_in_shape, dtype=dtype)
        ary_in[:, :, :] = np.nan

        for i in range(converted_df.shape[0]):
            ary_to_fil = converted_df[i]
            ary_in[i, :ary_to_fil.shape[0], :] = ary_to_fil

        ary_out_shape = converted_df.shape[0], sample_to_n_rows, n_fields
        ary_out = np.zeros(ary_out_shape, dtype=dtype)
        cls.scale_and_average_df_numba(ary_in, ary_out)

        return ary_out, index_map, discarded_fids

    @staticmethod
    def inverse_map(map):
        inv_map = {}
        for k, v in map.items():
            inv_map[v] = inv_map.get(v, [])
            inv_map[v].append(k)
        return inv_map


class VisualiseClustering:
    colorcycle = cycle(['C0', 'C1'])
    sizecycle = cycle([1, 0.1])
    crs = CRS(('epsg', '3857'))
    figsize = (10, 10)

    def __init__(self, airspace, save_path=None, intermediate_results=False, show=True, use_titles=True):
        self.airspace_projected = prepare_gdf_for_plotting(airspace)
        self.save_path = save_path
        self.intermediate_results = intermediate_results
        self.show = show
        self.use_titles = use_titles
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def __del__(self):
        plt.close()

    def intermediate_result(self, left, right, title, fname_arg=None):
        left_flat = left.reshape((-1, left.shape[2]))
        right_flat = right.reshape((-1, right.shape[2]))
        ax = self.ax
        self.airspace_projected.plot(ax=ax, alpha=0.5, edgecolor='k')
        ax.set_axis_off()
        colorcycle = cycle(['C2', 'C3'])
        for tracks in [left_flat, right_flat]:
            color = next(colorcycle)
            size = 0.1
            gs = geopandas.GeoSeries(geopandas.points_from_xy(tracks[:, 0], tracks[:, 1]))
            gs.crs = self.crs
            gs.plot(ax=ax, color=color, markersize=size, linewidth=size)
        if self.use_titles:
            ax.set_title(title)
        if self.save_path:
            self.fig.savefig(self.save_path.format(fname_arg))
            # self.intermediate_result_counter += 1
        if self.show:
            self.fig.show()
        self.ax.clear()

    def plot_cluster(self, tracks_concat, title, fname_arg=None, is_intermediate=False):
        if not self.intermediate_results and is_intermediate:
            return
        tracks_mean = tracks_concat.mean(axis=0)
        tracks_concat_flat = tracks_concat.reshape((-1, tracks_concat.shape[2]))
        ax = self.ax
        self.airspace_projected.plot(ax=ax, alpha=0.5, edgecolor='k')

        ax.set_axis_off()

        for tracks in [tracks_mean, tracks_concat_flat]:
            color = next(self.colorcycle)
            size = next(self.sizecycle)
            gs = geopandas.GeoSeries(geopandas.points_from_xy(tracks[:, 0], tracks[:, 1]))
            gs.crs = self.crs
            gs.plot(ax=ax, color=color, markersize=size, linewidth=size)
        # ax.scatter(tracks_mean[:, 0].min() - 1.2 * (tracks_mean[:, 0].max() - tracks_mean[:, 0].min()),
        #            tracks_mean[:, 1].min(), c='C2')
        if self.use_titles:
            ax.set_title(title)
        if self.save_path and fname_arg:
            self.fig.savefig(self.save_path.format(fname_arg))
        if self.show:
            self.fig.show()
        self.ax.clear()

    def plot_means(self, tracks, unclustered, title=None, fname_arg=None):

        ax = self.ax
        self.airspace_projected.plot(ax=ax, alpha=0.5, edgecolor='k')

        ax.set_axis_off()
        if tracks is not None:
            gs = geopandas.GeoSeries(geopandas.points_from_xy(tracks[:, 0], tracks[:, 1]))
            gs.crs = self.crs
            gs.plot(ax=ax, markersize=1, linewidth=1, color='C0')
        if unclustered is not None:
            gs_unclustered = geopandas.GeoSeries(geopandas.points_from_xy(unclustered[:, 0], unclustered[:, 1]))
            gs_unclustered.crs = self.crs
            gs_unclustered.plot(ax=ax, markersize=0.1, linewidth=0.1, color='C1')
        if self.use_titles:
            ax.set_title(title)
        if self.save_path and fname_arg:
            self.fig.savefig(self.save_path.format(fname_arg))
        if self.show:
            self.fig.show()
        self.ax.clear()


def get_non_diagonal_elements(square_matrix):
    n = square_matrix.shape[0]
    return np.lib.stride_tricks.as_strided(
        square_matrix, (n - 1, n + 1), (square_matrix.itemsize * (n + 1), square_matrix.itemsize))[:, 1:]


def when_everything_within_interval(W_ii, W):
    W_ii = get_non_diagonal_elements(W_ii)
    W = get_non_diagonal_elements(W)
    # if W_ii.shape[0] < 15:
    #     return True
    # return W.mean() > 0.4 and np.sum(W_ii.min(axis=1) < W.mean() - W.std()) < 2
    return W_ii.mean() - W_ii.std() > 0.28



if __name__ == "__main__":
    verbose = True
    airspace_query = "airport=='EHAM'"
    data_date = '20180101'
    minalt = 200  # ft
    maxalt = 10000
    e_mean = 0.4
    e_var = 1
    min_cluster_size = 10
    n_data_points = 200
    fields = ['lat', 'lon']
    K = 8
    plot_individual_clusters = True
    use_plot_titles = True
    show_plots = True
    # stop_function = lambda X, Y: np.mean(X) > .4
    stop_function = when_everything_within_interval

    one_matrix_shape = n_data_points, len(fields)
    airspace = ehaa_airspace.query(airspace_query)

    visualisation = VisualiseClustering(airspace, save_path='./figures/eham_{date}_{{0}}.png'.format(date=data_date),
                                        intermediate_results=plot_individual_clusters, use_titles=use_plot_titles,
                                        show=show_plots)
    machine_precision = np.finfo(np.float64).eps

    log = create_logger(verbose, "Clustering")
    log("Start reading csv")

    # #### SINGLE FILENAME
    df = pd.read_csv(
        'data/adsb_decoded_in_eham/ADSB_DECODED_{0}.csv.gz'.format(data_date))  # , converters={'callsign': lambda s: s.replace('_', '')})
    df.sort_values(by=['fid', 'ts'], inplace=True)

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

    x, fid_list, discarded_fids = Clustering.scale_and_average_df_numba_wrapper(df, n_data_points, fields,
                                                                                alt_conversion=3 * 1852 * 0.3048 / 2)

    log("Threw away {0} fid's that had less than {1} rows".format(len(discarded_fids), n_data_points))

    visualisation.plot_means(None, x.reshape((-1, x.shape[2])), title='All tracks', fname_arg='00_unclustered')
    log("Converted coordinates, Queued adjacency matrix calculation")

    clustering = Clustering(x, K, e_mean, e_var, min_cluster_size, stop_function=stop_function, visualisation=visualisation)
    W = clustering.calculate_adjacency()
    log("Calculated adjacency matrix")
    clustering.spectral_cluster(W)
    log("Finished spectral_cluster")
    cluster_result = clustering.result_indices
    n_clusters = clustering.n_clusters

    fid_to_cluster_map = {fid_list[i]: c_r for i, c_r in enumerate(cluster_result)}
    df.query('fid not in @discarded_fids', inplace=True)
    df['cluster'] = df['fid'].map(fid_to_cluster_map)
    log("Determined fid_to_cluster_map")
    log("{0} clusters found (e_mean={1}, e_var={2})".format(clustering.n_clusters, e_mean, e_var))

    cluster_to_fid_map = Clustering.inverse_map(fid_to_cluster_map)
    fid_to_index_map = {v: k for k, v in enumerate(fid_list)}
    tracks_means = []
    noise = None
    n_noise = None
    # Sort from largest to smallest cluster
    for key, fids in sorted(cluster_to_fid_map.items(), key=lambda a: len(a[1]))[::-1]:
        tracks_concat_index = np.array([fid_to_index_map[fid] for fid in fids])
        tracks_concat = x[tracks_concat_index]
        if key == -1:
            log("Couldn't cluster {0} tracks".format(len(tracks_concat_index)))
            n_noise = len(tracks_concat_index)
            noise = tracks_concat.reshape((-1, x.shape[2]))
            continue
        tracks_mean = tracks_concat.mean(axis=0)
        tracks_means.append(tracks_mean)
    visualisation.plot_means(np.vstack(tracks_means), noise,
                             title='Cluster means (blue) and unclustered tracks (orange)',
                             fname_arg='AA_results_and_noise')
    del visualisation
    with open('data/clustered/eham_{0}.csv'.format(data_date), 'w') as fp:
        parameters = {"K": K, "e_mean": e_mean, "e_var": e_var, "n_tracks_clustered": n_clusters,
                      "n_unclustered": n_noise}
        fp.write(repr(parameters))
        fp.write("\n")
        df.to_csv(fp)
