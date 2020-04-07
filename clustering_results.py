import ast
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statistics import radial_scatter_plot
from clustering import lat2y, lon2x
from nl_airspace_def import ehaa_airspace
from tools import create_logger
from tools import subsample_dataframe

if __name__ == "__main__":
    data_date = '20180101'
    n_data_points = 200
    with open('data/clustered/eham_{0}.csv'.format(data_date), 'r') as fp:
        parameters = ast.literal_eval(fp.readline())
        df = pd.read_csv(fp)
    df.sort_values(by=['cluster', 'ts'], inplace=True)
    airspace_query = "airport=='{}'".format("EHAM")
    airspace = ehaa_airspace.query(airspace_query)

    airspace_x = lon2x(airspace[airspace['type'] == 'CTR'].geometry.centroid.x).iloc[0]
    airspace_y = lat2y(airspace[airspace['type'] == 'CTR'].geometry.centroid.y).iloc[0]
    df['x'] = lon2x(df['lon'].values) - airspace_x
    df['y'] = lat2y(df['lat'].values) - airspace_y
    df['dist_to_airport'] = np.sqrt((df['x']) ** 2 + (df['y']) ** 2) / 1852
    df['FL'] = df['alt']/100
    df['trk_rad'] = np.deg2rad(df['trk'])
    tracks_per_cluster = df.groupby('cluster')['fid'].unique().apply(len)
    cluster_filenames = ['figures/' + fn for fn in os.listdir('figures') if fn[fn.rfind('_') + 1:-4].isdigit()]
    for cluster, n_tracks in tracks_per_cluster.sort_values(ascending=False).iteritems():
        print("Cluster {0} (n_tracks={1}".format(cluster, n_tracks))
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
        points.plot.scatter(x='index_along_track', y='alt', s=0.01)
        plt.show()
        plt.figure()
        points.plot.scatter(x='index_along_track', y='gs', s=0.01)
        plt.show()
        plt.figure()
        points.plot.scatter(x='index_along_track', y='trk', s=0.01)
        plt.show()
        plt.figure()
        points.plot.scatter(x='index_along_track', y='roc', s=0.01)
        plt.show()
        radial_scatter_plot(points, 'alt', 'h [ft]', query='roc <= 0', normalize_cmap=False, invert_theta=True)
        radial_scatter_plot(points, 'alt', 'h [ft]', query='roc >= 0', normalize_cmap=False, invert_theta=True)
        if input("Continue? [y/n]").upper() == 'N':
            break
