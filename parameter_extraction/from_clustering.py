import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

if __name__ == "__main__":
    data_date = '20180101'
    n_data_points = 200
    with open('data/clustered/eham_{0}.csv'.format(data_date), 'r') as fp:
        parameters = ast.literal_eval(fp.readline())
        df = pd.read_csv(fp)
    df.sort_values(by=['cluster', 'ts'], inplace=True)

    tracks_per_cluster = df.groupby('cluster')['fid'].unique().apply(len)

    for cluster, n_tracks in tracks_per_cluster.sort_values(ascending=False).iteritems():
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
        points_per_fid = points.set_index(['fid', 'index_along_track']).sort_index()[['x', 'y']].to_numpy().reshape(n_tracks, 200, 2)
        plt.figure()
        medoid_index = medoid(points_per_fid.reshape(n_tracks, -1), axis=0, indexonly=True)
        mean = points_per_fid.reshape(n_tracks, -1).mean(axis=0).reshape(200, 2)
        for i, trajectory in enumerate(points_per_fid):
            if i == medoid_index:
                plt.scatter(trajectory[:, 0], trajectory[:, 1], s=10, c='blue', label='Medoid')
            else:
                plt.scatter(trajectory[:, 0], trajectory[:, 1], s=0.1)
        plt.scatter(mean[:, 0], mean[:, 1], s=10, c='orange', label='Mean')
        plt.legend()
        plt.show()
        # plt.figure()
        # points.plot.scatter(y='index_along_track', x='gs', s=0.01)
        # plt.show()
        pass

