import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        plt.figure()
        points.plot.scatter(y='index_along_track', x='gs', s=0.01)
        plt.show()
        break

