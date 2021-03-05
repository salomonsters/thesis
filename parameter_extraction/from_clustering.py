import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas
from shapely.geometry import Point, LineString, shape


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


def count_intersections(a, b):
    if a['geometry'] == b['geometry']:
        return np.nan
    if not a['geometry'].intersects(b['geometry']):
        return 0
    intersection = a['geometry'].intersection(b['geometry'])

    if intersection.geom_type == 'Point':
        return 1
    elif intersection.geom_type == 'MultiPoint':
        return len(intersection)
    else:
        raise NotImplementedError("Unknown geom_type {}".format(intersection.geom_type))

if __name__ == "__main__":
    data_date = '20180101'
    n_data_points = 200
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
        points_per_fid = points.set_index(['fid', 'index_along_track']).sort_index()[['x', 'y', 'alt']].to_numpy().reshape(n_tracks, 200, 3)
        # plt.figure()
        # medoid_index = medoid(points_per_fid.reshape(n_tracks, -1), axis=0, indexonly=True)
        mean = points_per_fid.reshape(n_tracks, -1).mean(axis=0).reshape(n_data_points, 3)
        mean_2d_track = mean[:, :2]
        mean_alt = mean[:, 2]
        # for i, trajectory in enumerate(points_per_fid):
        #     if i == medoid_index:
        #         plt.scatter(trajectory[:, 0], trajectory[:, 1], s=10, c='blue', label='Medoid')
        #     else:
        #         plt.scatter(trajectory[:, 0], trajectory[:, 1], s=0.1)
        # plt.scatter(mean[:, 0], mean[:, 1], s=10, c='orange', label='Mean')
        # plt.legend()
        # plt.show()

        rows.append((cluster, n_tracks, LineString(mean), mean_2d_track, mean_alt))

        # plt.figure()
        # points.plot.scatter(y='index_along_track', x='gs', s=0.01)
        # plt.show()

    gdf = geopandas.GeoDataFrame.from_records(rows, columns=['cluster', 'n_tracks', 'geometry', 'mean_track', 'mean_alt'])
    intersection_map = np.array([[left.intersects(right) for left in gdf['geometry']] for right in gdf['geometry']])
    intersection_heatmap = np.array([[count_intersections(left, right) for _, left in gdf.iterrows()] for _, right in gdf.iterrows()])

    if plot_intersection_heatmap:
        import seaborn as sns
        ax = sns.heatmap(intersection_heatmap / 200 * 100, cbar_kws={"label": "%"})
        ax.set_title("% intra-cluster trajectory points intersecting per cluster pair")

        plt.show()



    # Todo tue: use high-overlap clusters to find method to find location/properties of intersection
    # Perhaps: identify consecutive points and use middle? And find non-intersecting points?
