import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clustering import lat2y, lon2x
from nl_airspace_def import ehaa_airspace

if __name__ == "__main__":
    data_date = '20180101'
    cluster_to_analyse = 48
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

    df = df.query('cluster==@cluster_to_analyse')
    tracks_in_cluster = len(df['fid'].unique())
    print("Analysing cluster {0} (n_tracks={1})".format(cluster_to_analyse, tracks_in_cluster))
    for fid, track_points in df.groupby('fid'):
        assert len(track_points) >= n_data_points
        resampling_indices = np.round(np.linspace(0, len(track_points) - 1, n_data_points)).astype('int')
        df.loc[track_points.index[resampling_indices], 'index_along_track'] = list(range(n_data_points))
        # Stupid SettingWithCopyWarning
        assert np.all(df.loc[track_points.index[resampling_indices], 'index_along_track'] >= 0)
    df_downsampled = df.query('index_along_track >= 0')
    points_along_track = df_downsampled.groupby('index_along_track')
    fields = ['x', 'y', 'alt', 'gs']
    df_along_track = points_along_track.mean()[fields].merge(points_along_track.quantile(q=0.0)[fields], suffixes=['', '_l'], right_index=True, left_index=True)
    df_along_track = df_along_track.merge(points_along_track.quantile(q=1)[fields], suffixes=['', '_u'], right_index=True, left_index=True)

    plt.figure()
    ax = df_along_track.plot(x='x', y='y')
    ax.fill_between(df_along_track['x_l'], df_along_track['y_l'], df_along_track['y_u'], alpha=0.5, step='mid', color='C0')
    ax.fill_between(df_along_track['x_u'], df_along_track['y_l'], df_along_track['y_u'], alpha=0.5, step='mid', color='C0')
    ax.fill_betweenx(df_along_track['y_l'], df_along_track['x_l'], df_along_track['x_u'], alpha=0.5, step='mid', color='C0')
    ax.fill_betweenx(df_along_track['y_u'], df_along_track['x_l'], df_along_track['x_u'], alpha=0.5, step='mid', color='C0')
    plt.legend(['mean', 'total area'])
    plt.show()

    plt.figure()
    ax = df_along_track.plot(y='alt',color='C3')
    ax.fill_between(df_along_track.index, df_along_track['alt_l'], df_along_track['alt_u'])
    plt.show()

    plt.figure()
    ax = df_along_track.plot(y='gs',color='C3')
    ax.fill_between(df_along_track.index, df_along_track['gs_l'], df_along_track['gs_u'])
    plt.show()

