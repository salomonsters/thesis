import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clustering import lat2y, lon2x
from nl_airspace_def import ehaa_airspace
from timesnapshots import cut_interval

if __name__ == "__main__":
    data_date = '20180101'
    clusters_to_analyse = [48, 51]
    n_data_points = 200
    with open('data/clustered/eham_{0}.csv'.format(data_date), 'r') as fp:
        parameters = ast.literal_eval(fp.readline())
        df_all = pd.read_csv(fp)
    df_all.sort_values(by=['cluster', 'ts'], inplace=True)
    airspace_query = "airport=='{}'".format("EHAM")
    airspace = ehaa_airspace.query(airspace_query)

    airspace_x = lon2x(airspace[airspace['type'] == 'CTR'].geometry.centroid.x).iloc[0]
    airspace_y = lat2y(airspace[airspace['type'] == 'CTR'].geometry.centroid.y).iloc[0]
    df_all['x'] = lon2x(df_all['lon'].values) - airspace_x
    df_all['y'] = lat2y(df_all['lat'].values) - airspace_y
    df_all['dist_to_airport'] = np.sqrt((df_all['x']) ** 2 + (df_all['y']) ** 2) / 1852
    df_all['FL'] = df_all['alt']/100
    df_all['trk_rad'] = np.deg2rad(df_all['trk'])
    for cluster_n in clusters_to_analyse:
        df = df_all.query('cluster==@cluster_n')
        # df_by_interval = cut_interval(df, 3600).reset_index().set_index(['interval', 'fid', 'ts'])
        # fid_per_interval = df_by_interval.groupby(by='interval')['fid'].unique().apply(len)
        tracks_in_cluster = len(df['fid'].unique())
        print("Analysing cluster(s) {0} (n_tracks={1})".format(cluster_n, tracks_in_cluster))
        fid_start_stop_times = []
        for fid, track_points in df.groupby('fid'):
            fid_start_stop_times.append([fid, track_points['ts'].min(), track_points['ts'].max()])
            assert len(track_points) >= n_data_points
            resampling_indices = np.round(np.linspace(0, len(track_points) - 1, n_data_points)).astype('int')
            df.loc[track_points.index[resampling_indices], 'index_along_track'] = list(range(n_data_points))
            # Stupid SettingWithCopyWarning
            assert np.all(df.loc[track_points.index[resampling_indices], 'index_along_track'] >= 0)
        df_start_stop_times = pd.DataFrame.from_records(fid_start_stop_times, columns=['fid', 't0', 'tend'])
        df_start_stop_times['duration'] = df_start_stop_times['tend'] - df_start_stop_times['t0']

        break
        #
        # df_downsampled = df.query('index_along_track >= 0')
        # points_along_track = df_downsampled.groupby('index_along_track')
        # fields = ['x', 'y', 'alt', 'gs']
        # df_along_track = points_along_track.mean()[fields].merge(points_along_track[fields].quantile(q=0.0), suffixes=['', '_l'], right_index=True, left_index=True)
        # df_along_track = df_along_track.merge(points_along_track[fields].quantile(q=1), suffixes=['', '_u'], right_index=True, left_index=True)
        #
        # plt.figure()
        # ax = df_along_track.plot(x='x', y='y')
        # ax.fill_between(df_along_track['x_l'], df_along_track['y_l'], df_along_track['y_u'], alpha=0.5, step='mid', color='C0')
        # ax.fill_between(df_along_track['x_u'], df_along_track['y_l'], df_along_track['y_u'], alpha=0.5, step='mid', color='C0')
        # ax.fill_betweenx(df_along_track['y_l'], df_along_track['x_l'], df_along_track['x_u'], alpha=0.5, step='mid', color='C0')
        # ax.fill_betweenx(df_along_track['y_u'], df_along_track['x_l'], df_along_track['x_u'], alpha=0.5, step='mid', color='C0')
        # plt.legend(['mean', 'total area'])
        # plt.show()
        #
        # plt.figure()
        # ax = df_along_track.plot(y='alt',color='C3')
        # ax.fill_between(df_along_track.index, df_along_track['alt_l'], df_along_track['alt_u'])
        # plt.show()
        #
        # plt.figure()
        # ax = df_along_track.plot(y='gs',color='C3')
        # ax.fill_between(df_along_track.index, df_along_track['gs_l'], df_along_track['gs_u'])
        # plt.show()
        #
