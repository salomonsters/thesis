import copy

import math
import matplotlib.colors
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from numba import cuda
from sklearn.cluster import DBSCAN, KMeans

from clustering import lat2y, lon2x
from nl_airspace_def import ehaa_airspace
from tools import create_logger
from tools import subsample_dataframe


def radial_scatter_plot(df_to_plot, field, field_label, query=None, invert_theta=False, normalize_cmap=False, cmap=None, columns=['trk_rad', 'dist_to_airport']):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    if query is not None:
        df = df_to_plot.query(query)
        ax.set_title(query)
    else:
        df = df_to_plot

    center_around_zero = None
    if cmap is None:
        cmap='cividis'
    if normalize_cmap:
        center_around_zero = matplotlib.colors.DivergingNorm(vmin=df[field].min(), vcenter=0., vmax=df[field].max())
        cmap = 'bwr'
    c = ax.scatter(df[columns[0]], df[columns[1]], c=df[field], s=0.1, cmap=cmap, alpha=0.75, norm=center_around_zero)
    ax.set_theta_zero_location('N')
    ax.set_thetagrids(list(np.arange(0, 360, 30)))
    ax.set_rmax(df.quantile(0.90)[columns[1]]*1.1)
    if invert_theta:
        ax.set_theta_direction(-1)
    colorbar = fig.colorbar(c)
    colorbar.set_label(field_label)

    plt.show()

USE_64 = False

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32


@cuda.jit("void(float{0}[:, :], int64[:], int64[:, :])".format(bits))
def distance_matrix(mat, counter, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m and i != j and i < j:
        # 3 nm, 10 FL's
        if math.fabs(mat[i, 0] - mat[j, 0]) < 3 and \
                math.fabs(mat[i, 2] - mat[j, 2]) < 10:  # 3 nm
            loc = cuda.atomic.add(counter, 0, 1)
            if loc >= out.shape[0]:
                cuda.atomic.add(counter, 1, 1)
            else:
                out[loc, 0] = i
                out[loc, 1] = j

        # for k in range(n):
        #     tmp = mat[i, k] - mat[j, k]
        #     d += tmp * tmp
        # if d < 2:
        #     out[i, j] = 1
        # else:
        #     out[i, j] = 0


def gpu_dist_matrix(mat):
    rows = mat.shape[0]

    block_dim = (32, 32)
    grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))
    counter = cuda.to_device(np.zeros(2, dtype=np.int64))
    nonzero_locations = cuda.device_array((X.shape[0]**2, 2), dtype=np.int64)
    stream1 = cuda.stream()
    stream2 = cuda.stream()
    mat2 = cuda.to_device(mat, stream=stream1)
    distance_matrix[grid_dim, block_dim](mat2, counter, nonzero_locations)
    out = nonzero_locations.copy_to_host(stream=stream2)
    stream2.synchronize()
    h_counter = counter.copy_to_host()
    print(h_counter)
    return out


def apply_clustering(method, cluster_kwargs, df, columns, cluster_offset_n=0):
    X = df[columns].to_numpy()
    if method == 'DBSCAN':
        clusters = DBSCAN(n_jobs=-2, **cluster_kwargs).fit_predict(X)
    elif method == 'KMeans':
        clusters = KMeans(n_jobs=-2, **cluster_kwargs).fit_predict(X)
    log("Clustering complete")
    df['cluster'] = np.where(clusters ==-1, -1, clusters + cluster_offset_n)
    return df


if __name__ == "__main__":
    show_stats_plots = False
    method = 'DBSCAN'
    cluster_kwargs = {'eps': 900, 'min_samples': 150}
    # method = 'KMeans'
    # cluster_kwargs = {'n_clusters': 8}
    verbose = True
    airport = 'EHAM'
    airspace_query = "airport=='{}'".format(airport)
    zoom = 15
    minalt = 200  # ft
    maxalt = 10000

    fields = ['lat', 'lon']

    airspace = ehaa_airspace.query(airspace_query)
    airspace_x = lon2x(airspace[airspace['type'] == 'CTR'].geometry.centroid.x).iloc[0]
    airspace_y = lat2y(airspace[airspace['type'] == 'CTR'].geometry.centroid.y).iloc[0]

    log = create_logger(verbose, "Statistics")


    log("Start reading csv")

    # #### SINGLE FILENAME
    df = pd.read_csv('data/adsb_decoded_in_{0}/ADSB_DECODED_20180101.csv.gz'.format(airport))  # , converters={'callsign': lambda s: s.replace('_', '')})

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

    df['x'] = lon2x(df['lon'].values) - airspace_x
    df['y'] = lat2y(df['lat'].values) - airspace_y
    df['trk_rad'] = np.deg2rad(df['trk'])
    # from scipy.spatial.distance import euclidean
    @numba.njit(parallel=True)
    def within_separation_minima(x1, x2):
        """
        Take to vectors and compute the distance between them. Vectors must be [x, y, alt]
        :param x1: vector
        :param x2: vector
        :return: 0 if within separation minimums (horizontal <3nm and alt <1000), else 2
        """
        hdist_sq = ((x1[:2]-x2[:2])**2).sum()
        vdist = np.abs(x1[2]-x2[2])
        if hdist_sq <= 9. and vdist <= 1000.:
            return 0
        else:
            return 2



    df['dist_to_airport'] = np.sqrt((df['x'])**2 + (df['y'])**2)/1852
    df['FL'] = df['alt']/100
    if show_stats_plots:
        plt.figure()
        df.plot.scatter(x='dist_to_airport', y='alt', s=0.01)
        plt.show()
        plt.figure()
        df.plot.scatter(x='dist_to_airport', y='gs', s=0.01)
        plt.show()
        plt.figure()
        df.plot.scatter(x='dist_to_airport', y='trk', s=0.01)
        plt.show()
        plt.figure()
        df.plot.scatter(x='dist_to_airport', y='roc', s=0.01)
        plt.show()
        radial_scatter_plot(df, 'alt', 'h [ft]', query='roc <= 0', normalize_cmap=False, invert_theta=True)
        radial_scatter_plot(df, 'alt', 'h [ft]', query='roc >= 0', normalize_cmap=False, invert_theta=True)

    # columns = ['trk', 'dist_to_airport', 'alt']

    # for bearing use atan2(x,y)!
    df['bearing'] = (np.arctan2(df['x'], df['y']) + 2 * np.pi) % (2 * np.pi)
    plot_columns = ['bearing', 'dist_to_airport']
    df['normalized_alt'] = df['alt']/2
    columns = ['x', 'y', 'normalized_alt']
    df_for_cluster = df.dropna(subset=columns).query('roc <= 0')
    timeCol = 'ts'
    delta_t = 3
    radial_scatter_plot(df_for_cluster, 'alt', 'h [ft]', query='roc <= 0', normalize_cmap=False, invert_theta=True, columns=plot_columns)
    log("Starting resampling to 1 datapoint per {0} seconds".format(delta_t))
    # Creation of the time sampling

    df_resampled = subsample_dataframe(df_for_cluster, delta_t)

    log("Finished resampling")
    radial_scatter_plot(df_resampled, 'alt', 'h [ft]', query='roc <= 0', normalize_cmap=False, invert_theta=True, columns=plot_columns)

    df_clustered = apply_clustering(method, cluster_kwargs, df_resampled, columns)
    # apply clustering but with bigger epsilon
    cluster_offset_n = max(df_clustered['cluster'])
    successful_cluster_selector = df_clustered['cluster'] != -1
    cluster_kwargs['eps'] = 2*1852
    # perhaps do DBscan where if one point of track is in cluster, all points of that track are in the cluster?
    # df_clustered = pd.concat([apply_clustering(method, cluster_kwargs, df_clustered[~successful_cluster_selector], columns, cluster_offset_n),
                              # df_clustered[successful_cluster_selector]])

    log("Number of clusters: {}".format(max(df_clustered['cluster'])))

    for i in sorted(set(df_clustered['cluster'])):
        radial_scatter_plot(df_resampled, 'alt', 'alt', query='cluster == {0}'.format(i), normalize_cmap=False, invert_theta=True, columns=plot_columns)
    # log("Starting GPU distance matrix computation")
    # D = gpu_dist_matrix(X)
    # log("Finished GPU distance matrix computation")
    # D = D[D.nonzero()[0][::2],:]
    # S = ss.eye(X.shape[0], format='coo') + ss.coo_matrix((np.ones(D.shape[0]), (D[:,0], D[:,1])), shape=(X.shape[0], X.shape[0]))

    #
    # import sklearn
    # log("Starting computation")
    # D = sklearn.metrics.pairwise_distances_chunked(X, n_jobs=-1)
    # log("Finished computation with n_jobs=-1")
    # i = 0
    # while True:
    #     try:
    #         n = next(D)
    #         i += 1
    #         log("i={0}, shape = {1}".format(i, n.shape))
    #     except StopIteration:
    #         log("Finished")
    #         break
    # log("Finished")
    #
    # print(D.shape)
    # print(D[-1,:])