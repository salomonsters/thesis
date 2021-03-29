import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas
from shapely.geometry import Point, LineString, shape
import itertools

from conflicts.simulate import Aircraft
S_h = Aircraft.horizontal_separation_requirement * 1852
S_v = Aircraft.vertical_separation_requirement

# From https://github.com/daleroberts/hdmedians/blob/master/hdmedians/medoid.py
def medoid(a, axis=1, indexonly=False):
    """
    Compute the medoid along the specified axis.
    Returns the medoid of the array elements.
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int
        Axis along which the medoid is computed. The default
        is to compute the median along the last axis of the array.
    indexonly : bool, optional
        If this is set to True, only the index of the medoid is returned.
    Returns
    -------
    medoid : ndarray or int
    """
    if axis == 1:
        diff = a.T[:, None, :] - a.T
        ssum = np.einsum('ijk,ijk->ij', diff, diff)
        idx = np.argmin(np.sum(np.sqrt(ssum), axis=1))
        if indexonly:
            return idx
        else:
            return a[:, idx]

    if axis == 0:
        diff = a[:, None, :] - a
        ssum = np.einsum('ijk,ijk->ij', diff, diff)
        idx = np.argmin(np.sum(np.sqrt(ssum), axis=1))
        if indexonly:
            return idx
        else:
            return a[idx, :]

    raise IndexError("axis {} out of bounds".format(axis))

def consecutive_string_lengths(condition):
    # https://stackoverflow.com/a/24343375
    return np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2]



def count_intersections(left, right):
    v_diff = left.mean_alt - right.mean_alt
    if not np.any(within_S_v := np.abs(v_diff) <= S_v):
        return 0

    h_distances = np.linalg.norm(left.mean_track - right.mean_track, axis=1)

    if not np.any(within_S_h := h_distances <= S_h):
        return 0
    consecutive_lengths = consecutive_string_lengths(within_S_h & within_S_v)

    return len(consecutive_lengths)

    # return np.sum(within_S_h & within_S_v)


    # a, b = left, right
    # if a['geometry'] == b['geometry']:
    #     return np.nan
    # if not a['geometry'].intersects(b['geometry']):
    #     return 0
    # intersection = a['geometry'].intersection(b['geometry'])
    #
    # if intersection.geom_type == 'Point':
    #     return 1
    # elif intersection.geom_type == 'MultiPoint':
    #     return len(intersection)
    # else:
    #     raise NotImplementedError("Unknown geom_type {}".format(intersection.geom_type))

if __name__ == "__main__":
    data_date = '20180101'
    n_data_points = 200
    columns = ['x', 'y', 'alt', 'gs', 'trk', 'roc']
    plot_intersection_heatmap = True
    with open('data/clustered/eham_{0}.csv'.format(data_date), 'r') as fp:
        parameters = ast.literal_eval(fp.readline())
        df = pd.read_csv(fp)
    df.sort_values(by=['cluster', 'ts'], inplace=True)

    tracks_per_cluster = df.groupby('cluster')['fid'].unique().apply(len)
    rows = []
    # for cluster, n_tracks in tracks_per_cluster.sort_values(ascending=False).iteritems():
    for cluster, n_tracks in tracks_per_cluster.iteritems():
        # Don't use unclustered tracks for now
        if cluster == -1:
            continue

        cluster_points = df[df['cluster'] == cluster]
        cluster_points['index_along_track'] = -1
        assert n_tracks == len(cluster_points['fid'].unique())
        for fid, track_points in cluster_points.groupby(by='fid'):
            assert len(track_points) >= n_data_points
            resampling_indices = np.round(np.linspace(0, len(track_points) - 1, n_data_points)).astype('int')
            cluster_points.loc[track_points.index[resampling_indices], 'index_along_track'] = list(range(n_data_points))
            # Stupid SettingWithCopyWarning
            assert np.all(cluster_points.loc[track_points.index[resampling_indices], 'index_along_track'] >= 0)
        points = cluster_points.query('index_along_track >= 0')
        points_per_fid = points.set_index(['fid', 'index_along_track']).sort_index()[columns].to_numpy().reshape(n_tracks, 200, len(columns))
        # plt.figure()
        # medoid_index = medoid(points_per_fid.reshape(n_tracks, -1), axis=0, indexonly=True)
        mean = points_per_fid.reshape(n_tracks, -1).mean(axis=0).reshape(n_data_points, len(columns))
        mean_2d_track = mean[:, :2]
        mean_alt = mean[:, 2]
        if mean.shape[1] > 2:
            other_states = mean[:, 3:]
        else:
            other_states = None
        # for i, trajectory in enumerate(points_per_fid):
        #     if i == medoid_index:
        #         plt.scatter(trajectory[:, 0], trajectory[:, 1], s=10, c='blue', label='Medoid')
        #     else:
        #         plt.scatter(trajectory[:, 0], trajectory[:, 1], s=0.1)
        # plt.scatter(mean[:, 0], mean[:, 1], s=10, c='orange', label='Mean')
        # plt.legend()
        # plt.show()

        rows.append((cluster, n_tracks, LineString(mean_2d_track), mean_2d_track, mean_alt, other_states))

        # plt.figure()
        # points.plot.scatter(y='index_along_track', x='gs', s=0.01)
        # plt.show()

    gdf = geopandas.GeoDataFrame.from_records(rows, columns=['cluster', 'n_tracks', 'geometry', 'mean_track', 'mean_alt', 'other_states'])
    intersection_map = np.array([[left.intersects(right) for left in gdf['geometry']] for right in gdf['geometry']])
    intersection_heatmap = np.array([[count_intersections(left, right) for _, left in gdf.iterrows()] for _, right in gdf.iterrows()])

    if plot_intersection_heatmap:
        import seaborn as sns
        ax = sns.heatmap(intersection_heatmap, cbar_kws={"label": "%"})
        ax.set_title("Number of distinct overlapping 'strings'")

        plt.show()



    # Todo tue: use high-overlap clusters to find method to find location/properties of intersection
    # Perhaps: identify consecutive points and use middle? And find non-intersecting points?
