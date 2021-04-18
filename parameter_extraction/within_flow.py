import ast
import warnings

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from scipy.stats import circmean, circstd

from conflicts.simulate import Aircraft

#G. May (1971). A Method for Predicting the Number of Near Mid-Air Collisions in a Defined Airspace.
# Operational Research Quarterly (1970-1977), 22(3), 237–251.         doi:10.2307/3007993
def expected_collisions_in_stream(row):
    gs = row['points_per_fid'][:, :, columns.index('gs')]
    N = gs.shape[0]
    assert gs.shape[1] == 200
    C_exp_list = []
    lam = 3600/row['mean_time_between_activations']
    for i in range(N):
        mean_gs_excluding_i = np.nanmean(np.delete(gs, i, axis=0), axis=1)
        mean_gs = np.nanmean(gs[i,:])
        gs_diff = np.nanmean(np.abs(mean_gs_excluding_i - mean_gs))
        C_exp = gs_diff*lam/np.nanmean(mean_gs_excluding_i)
        C_exp_list.append(C_exp)
    return np.mean(C_exp_list)



if __name__ == "__main__":
    S_h_in_nm = 6
    S_h = 1852 * S_h_in_nm
    S_v = 1000
    t_l = 10./60.
    timeshift = None
    as_events = True
    use_weighted_least_squares = False
    calculate_V_rel_method = 'first'  # 'closest' or 'first'

    timeshift_suffix = ''
    filename_prefix = 'eham_stop_mean_0.25_std_0.25'
    t_col = 'ts'
    if timeshift is not None:
        timeshift_suffix = '-timeshift-uniform-0-{}'.format(timeshift)
    elif as_events:
        timeshift_suffix = '-as-events-repeats-5'

    replay_results_file = 'data/conflict_replay_results/{}_20180101-20180102-20180104-20180105-splits_[0-1-2-3]-S_h-{}-S_v-{}-t_l-{:.4f}{}.xlsx'.format(filename_prefix, S_h_in_nm, S_v, t_l, timeshift_suffix)
    replay_df_all_splits = pd.read_excel(replay_results_file)
    combined_df_list = []
    for split in range(4):
        data_date = '20180101-20180102-20180104-20180105_split_{}'.format(split)
        replay_results_query = 'split==@split & type=="within"'
        replay_df = replay_df_all_splits.query(replay_results_query)[['i', 'j', 'type', 'unclustered', 'split', 'conflicts']]
        # Replay df has (i, nan) for within-cluster conflicts but gdf will have (i, i)
        replay_df = replay_df.assign(j=np.where(np.isnan(replay_df['j']), replay_df['i'], replay_df['j']).astype(int))

        replay_df.set_index(['i'], inplace=True)
        replay_df.sort_index(inplace=True)
        n_data_points = 200
        columns = ['x', 'y', 'alt', 'gs', 'trk', 'roc']
        other_states_gs_index = columns.index('gs') - 3
        other_states_trk_index = columns.index('trk') - 3


        with open('data/clustered/{2}_{0}{1}.csv'.format(data_date, timeshift_suffix, filename_prefix), 'r') as fp:
            parameters = None
            if not as_events:
                parameters = ast.literal_eval(fp.readline())
            df = pd.read_csv(fp)
        df['cluster'] = np.where(df['cluster']==-1, 0, df['cluster'])
        df.sort_values(by=['cluster', t_col], inplace=True)
        df[t_col] = df[t_col] - df[t_col].min()

        tracks_per_cluster = df.groupby('cluster')['fid'].unique().apply(len)
        rows = []

        for cluster, n_tracks in tracks_per_cluster.iteritems():

            cluster_points = df.query('cluster== @cluster').assign(index_along_track=-1)
            assert n_tracks == len(cluster_points['fid'].unique())
            cluster_ts_0_items = []
            for fid, track_points in cluster_points.groupby(by='fid'):
                if len(track_points) < n_data_points:
                    # warnings.warn("{} has has {} datapoints, which is less than {}".format(fid, len(track_points), n_data_points))
                    n_tracks -= 1
                    continue
                assert len(track_points) >= n_data_points
                resampling_indices = np.round(np.linspace(0, len(track_points) - 1, n_data_points)).astype('int')
                cluster_points.loc[track_points.index[resampling_indices], 'index_along_track'] = list(range(n_data_points))
                cluster_ts_0_items.append(track_points[t_col].min())
            points = cluster_points.query('index_along_track >= 0')
            points_per_fid = points.set_index(['fid', 'index_along_track']).sort_index()[columns].to_numpy().reshape(
                n_tracks, 200, len(columns))

            mean = np.nanmean(points_per_fid.reshape(n_tracks, -1), axis=0).reshape(n_data_points, len(columns))
            if 'trk' in columns:
                trk_mean = circmean(points_per_fid[:, :, columns.index('trk')], high=360, axis=0, nan_policy='omit')
                mean[:, columns.index('trk')] = trk_mean
            mean_2d_track = mean[:, :2]
            mean_alt = mean[:, 2]
            if mean.shape[1] > 2:
                other_states = mean[:, 3:]
            else:
                other_states = None
            mean_time_between_activations = np.diff(np.sort(cluster_ts_0_items))[1:].mean()
            hourly_arrival_rates = np.histogram(np.array(cluster_ts_0_items) / 3600, bins=range(24))[0]

            active_duration_in_hours = (np.max(cluster_ts_0_items) - np.min(cluster_ts_0_items)) / 3600

            rows.append((cluster, n_tracks, mean_2d_track, mean_alt, other_states, points_per_fid,
                         active_duration_in_hours, hourly_arrival_rates, mean_time_between_activations))

            # plt.figure()
            # points.plot.scatter(y='index_along_track', x='gs', s=0.01)
            # plt.show()

        gdf = pd.DataFrame.from_records(rows,
                                        columns=['cluster', 'n_tracks', 'mean_track', 'mean_alt',
                                                 'other_states', 'points_per_fid', 'active_duration_in_hours',
                                                 'hourly_arrival_rates', 'mean_time_between_activations'])
        # gdf_joined = gdf.merge(gdf, suffixes=['_i', '_j'], how='cross')
        # gdf_joined = gdf_joined.rename({'cluster_i': 'i', 'cluster_j': 'j'}, axis=1).set_index(['i', 'j'])

        combined_df = replay_df.join(gdf.set_index(['cluster']))
        combined_df.reset_index(inplace=True)
        combined_df_list.append(combined_df)

    df_with_geometry = pd.concat(combined_df_list).reset_index(drop=False)
    del combined_df_list, combined_df, gdf, rows, replay_df
    df = df_with_geometry[['i', 'unclustered', 'split', 'conflicts', 'n_tracks', 'active_duration_in_hours', 'mean_time_between_activations']]

    df = df.assign(conflicts_normalised=df['conflicts'] / df['n_tracks'])
    df = df.assign(
        conflicts_normalised_time_corrected=df['conflicts_normalised'] * 3600 / df['mean_time_between_activations'],
        V_mean=df_with_geometry.apply(lambda row: np.nanmean(row['points_per_fid'][:, :, columns.index('gs')], axis=0).mean(), axis=1),
        V_std=df_with_geometry.apply(lambda row: np.nanstd(row['points_per_fid'][:, :, columns.index('gs')], axis=0).mean(), axis=1),
        V_std_max=df_with_geometry.apply(lambda row: np.nanstd(row['points_per_fid'][:, :, columns.index('gs')], axis=0).max(), axis=1),
        V_mean_max=df_with_geometry.apply(lambda row: np.nanmax(row['points_per_fid'][:, :, columns.index('gs')][None, :] - row['points_per_fid'][:, :, columns.index('gs')][:, None], axis=2).mean(), axis=1),
        trk_std=df_with_geometry.apply(lambda row: circstd(row['points_per_fid'][:, :, columns.index('trk')], high=360, axis=0, nan_policy='omit').mean(), axis=1),
        E_NMAC=df_with_geometry.apply(expected_collisions_in_stream, axis=1)
    )
    df['sin_trk_std_halved'] = np.sin(np.radians(df['trk_std']/2))
    df['cos_trk_std'] = np.cos(np.radians(df['trk_std']))
    from from_clustering import find_and_plot_correlation
    fig, ax = plt.subplots()
    find_and_plot_correlation(df.query('conflicts_normalised_time_corrected<10'), 'conflicts_normalised_time_corrected', 'E_NMAC', ax)
    # X, Y = df.query('conflicts_normalised_time_corrected<10')[['conflicts_normalised_time_corrected', 'E_NMAC']].to_numpy().T
    # df.query('conflicts_normalised_time_corrected<10').plot.scatter('conflicts_normalised_time_corrected', 'E_NMAC')
    plt.show()