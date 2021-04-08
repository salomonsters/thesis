import ast

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from conflicts.simulate import Aircraft

#S_h = Aircraft.horizontal_separation_requirement * 1852
#S_v = Aircraft.vertical_separation_requirement


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
    where_array = np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0]
    return np.diff(where_array)[::2]

def overlap_type(condition):
    # https://stackoverflow.com/a/24343375
    where_array = np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0]
    assert where_array.shape[0] > 1
    if where_array.shape[0] == 2 and np.all(condition):
        return 'all'
    elif where_array[0] == 0 and where_array.shape[0] == 3 and where_array[2] == n_data_points: # where_array[1] > n_data_points*0.15:
        return 'begin'

    return 'other'

    # return where_array[0]
    # elif where_array.shape[0] % 2 == 0 and where_array[-2] > n_data_points*0.5 and where_array[-1] == n_data_points:
    #     if where_array.shape[0] == 2:
    #         return 'merge'
    #         # return np.atleast_1d(where_array[-2])
    #     else:
    #         return 'multiple_cross'
    # else:
    #     return 'single_cross'
    #     # return where_array[:-1]

def overlap(left, right):
    if left['cluster'] == right['cluster']:
        return 'same'
    v_diff = left.mean_alt - right.mean_alt
    if not np.any(within_S_v := np.abs(v_diff) <= S_v):
        return 'none'

    h_distances = np.linalg.norm(left.mean_track - right.mean_track, axis=1)

    if not np.any(within_S_h := h_distances <= S_h) or not np.any(within_S_h & within_S_v):
        return 'none'
    # consecutive_lengths = consecutive_string_lengths(within_S_h & within_S_v)
    return overlap_type(within_S_h & within_S_v)
    #return len(consecutive_lengths)

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

def calculate_V_rel_corrected_at(row, at):
    """
    Calculates sqrt(V_i**2+V_j**2-2*V_i*V_j*cos(trk_diff))/sin(trk_diff)/V_i/V_j*lambda_i_max*lambda_j_max
    :param row:
    :param at:
    :return:
    """
    gs_i = row['other_states_i'][at, other_states_gs_index]
    gs_j = row['other_states_j'][at, other_states_gs_index]
    trk_i = row['other_states_i'][at, other_states_trk_index]
    trk_j = row['other_states_j'][at, other_states_trk_index]
    trk_diff = np.radians(np.abs(trk_i-trk_j))
    if trk_diff > np.pi:
        trk_diff = trk_diff - np.pi
    V_rel = np.sqrt(gs_i**2+gs_j**2-2*gs_i*gs_j*np.cos(trk_diff))
    hourly_arrival_rates_prod = row['hourly_arrival_rates_i'] * row['hourly_arrival_rates_j']
    nonzero_hourly_arrival_rates_prod = hourly_arrival_rates_prod[hourly_arrival_rates_prod > 0]
    lam_correction = nonzero_hourly_arrival_rates_prod.mean()
    V_rel_corr = V_rel * lam_correction/(gs_i*gs_j*np.sin(trk_diff))
    return V_rel_corr



if __name__ == "__main__":
    S_h_in_nm = 3
    S_h = 1852 * S_h_in_nm
    S_v = 1000
    use_weighted_least_squares = True

    replay_results_file = 'data/conflict_replay_results/eham_stop_mean_std_0.28_20180101-20180102-20180104-20180105-splits_[0-1-2-3]-S_h-{}-S_v-{}-t_l-0.1667.xlsx'.format(S_h_in_nm, S_v)
    replay_df_all_splits = pd.read_excel(replay_results_file)
    combined_df_list = []
    for split in range(4):
        data_date = '20180101-20180102-20180104-20180105_split_{}'.format(split)
        replay_results_query = 'split==@split'
        replay_df = replay_df_all_splits.query(replay_results_query)
        # Replay df has (i, nan) for within-cluster conflicts but gdf will have (i, i)
        replay_df = replay_df.assign(j=np.where(np.isnan(replay_df['j']), replay_df['i'], replay_df['j']).astype(int))
        replay_df.set_index(['i', 'j'], inplace=True)
        replay_df.sort_index(inplace=True)
        n_data_points = 200
        columns = ['x', 'y', 'alt', 'gs', 'trk', 'roc']
        other_states_gs_index = columns.index('gs') - 3
        other_states_trk_index = columns.index('trk') - 3
        # plot_intersection_heatmap = True
        with open('data/clustered/eham_stop_mean_std_0.28_{0}.csv'.format(data_date), 'r') as fp:
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

            cluster_points = df.query('cluster== @cluster')
            cluster_points = cluster_points.assign(index_along_track=-1)
            assert n_tracks == len(cluster_points['fid'].unique())
            cluster_ts_0_items = []
            for fid, track_points in cluster_points.groupby(by='fid'):
                assert len(track_points) >= n_data_points
                resampling_indices = np.round(np.linspace(0, len(track_points) - 1, n_data_points)).astype('int')
                cluster_points.loc[track_points.index[resampling_indices], 'index_along_track'] = list(range(n_data_points))
                # # Stupid SettingWithCopyWarning
                # assert np.all(cluster_points.loc[track_points.index[resampling_indices], 'index_along_track'] >= 0)
                cluster_ts_0_items.append(track_points['ts'].min())
            points = cluster_points.query('index_along_track >= 0')
            points_per_fid = points.set_index(['fid', 'index_along_track']).sort_index()[columns].to_numpy().reshape(n_tracks, 200, len(columns))
            # plt.figure()
            # medoid_index = medoid(points_per_fid.reshape(n_tracks, -1), axis=0, indexonly=True)
            mean = np.nanmean(points_per_fid.reshape(n_tracks, -1), axis=0).reshape(n_data_points, len(columns))
            mean_2d_track = mean[:, :2]
            mean_alt = mean[:, 2]
            if mean.shape[1] > 2:
                other_states = mean[:, 3:]
            else:
                other_states = None

            # hourly_arrivals = np.histogram(arrival_times := (cluster_ts_0_items - np.min(cluster_ts_0_items))/3600, bins=range(int(np.max(arrival_times))))[0].mean()
            hourly_arrival_rates = np.histogram(np.array(cluster_ts_0_items)/3600, bins=range(24))[0]

            active_duration_in_hours = (np.max(cluster_ts_0_items) - np.min(cluster_ts_0_items))/3600
            t_start, t_end = np.min(cluster_ts_0_items), np.max(cluster_ts_0_items)
            # for i, trajectory in enumerate(points_per_fid):
            #     if i == medoid_index:
            #         plt.scatter(trajectory[:, 0], trajectory[:, 1], s=10, c='blue', label='Medoid')
            #     else:
            #         plt.scatter(trajectory[:, 0], trajectory[:, 1], s=0.1)
            # plt.scatter(mean[:, 0], mean[:, 1], s=10, c='orange', label='Mean')
            # plt.legend()
            # plt.show()

            rows.append((cluster, n_tracks, LineString(mean_2d_track), mean_2d_track, mean_alt, other_states, active_duration_in_hours, t_start, t_end, hourly_arrival_rates))

            # plt.figure()
            # points.plot.scatter(y='index_along_track', x='gs', s=0.01)
            # plt.show()

        gdf = geopandas.GeoDataFrame.from_records(rows, columns=['cluster', 'n_tracks', 'geometry', 'mean_track', 'mean_alt', 'other_states', 'active_duration_in_hours', 't_start', 't_end','hourly_arrival_rates'])
        gdf['hourly_arrivals'] = gdf['n_tracks']/gdf['active_duration_in_hours']
        # intersection_map = np.array([[left.intersects(right) for left in gdf['geometry']] for right in gdf['geometry']])
        intersection_heatmap = np.array([[overlap(left, right) for _, left in gdf.iterrows()] for _, right in gdf.iterrows()], dtype='object')
        intersection_heatmap_extra_row_and_col = np.zeros(shape=np.array(intersection_heatmap.shape) + [1,1], dtype='object')
        intersection_heatmap_extra_row_and_col[:, :] = np.nan
        intersection_heatmap_extra_row_and_col[1:, 1:] = intersection_heatmap
        # if plot_intersection_heatmap:
        #     import seaborn as sns
        #     ax = sns.heatmap(intersection_heatmap, cbar_kws={"label": "%"})
        #     ax.set_title("Number of distinct overlapping 'strings'")
        #
        #     plt.show()

        gdf_joined = gdf.merge(gdf, suffixes=['_i', '_j'], how='cross')
        gdf_joined = gdf_joined.rename({'cluster_i': 'i', 'cluster_j': 'j'}, axis=1).set_index(['i', 'j'])

        combined_df = replay_df.join(gdf_joined)
        combined_df.reset_index(inplace=True)
        combined_df['overlap_type'] = combined_df.apply(lambda row: intersection_heatmap_extra_row_and_col[row['i'], row['j']], axis=1)
        combined_df.query('i>0 and j>0', inplace=True)
        combined_df['track_product'] = combined_df['n_tracks_i'] * combined_df['n_tracks_j']
        combined_df['combined_active_hrs'] = combined_df.apply(lambda x: np.max(x[['t_end_i', 't_end_j']]) -  np.min(x[['t_start_i', 't_start_j']]), axis=1)/3600
        combined_df['conflicts_per_active_hr'] = combined_df['conflicts']/combined_df['combined_active_hrs']

        # combined_df.plot.hexbin(x='hourly_arrivals_i', y='hourly_arrivals_j', C='conflicts')
        # plt.show()

        # Todo tue: use high-overlap clusters to find method to find location/properties of intersection
        # Perhaps: identify consecutive points and use middle? And find non-intersecting points?

        combined_df['V_rel_corr'] = np.nan
        for i in combined_df.index:
            ovl_type = combined_df.loc[i]['overlap_type']
            if isinstance(ovl_type, np.ndarray):
                shape = ovl_type.shape[0]
                if shape <= 2:
                    at = int(ovl_type.mean())
                    V_rel_corr = calculate_V_rel_corrected_at(combined_df.loc[i], ovl_type[0])
                    combined_df.loc[i, 'V_rel_corr'] = V_rel_corr
                else:
                    pass#print(ovl_type)
            pass
        combined_df['conflicts_predicted'] = 2*S_h/1852*combined_df['V_rel_corr']#*combined_df['hourly_arrivals_i']*combined_df['hourly_arrivals_j']
        combined_df_list.append(combined_df)
    combined_df_all = pd.concat(combined_df_list)
    # combined_df_all.query('conflicts_predicted<3', inplace=True)
    # high_conflicts = combined_df.query('type=="between"').sort_values('conflicts_per_active_hr')[-100:]
    fig, ax = plt.subplots(2, 1)
    combined_df_all.groupby('overlap_type')['conflicts'].sum().plot.barh(ax=ax[0])
    ax[0].set_xlabel("Sum of conflicts")
    combined_df_all.boxplot(column='conflicts_per_active_hr', by='overlap_type', vert=False, ax=ax[1])
    # plt.subplots_adjust(left=0.2)
    ax[1].set_xlabel("Conflicts_per_active_hr [1/h]")
    plt.show()
    if False:
        from sklearn.linear_model import LinearRegression
        X = combined_df_all['conflicts_predicted'].values.reshape(-1, 1)
        Y = combined_df_all['conflicts_per_active_hr'].values.reshape(-1, 1)
        W = combined_df_all['track_product'].values.reshape(-1, 1)
        non_nan_values = ~pd.isna(X+Y)
        linear_regressor = LinearRegression()
        if use_weighted_least_squares:
            linear_regressor.fit(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1), sample_weight=W[non_nan_values].ravel())
            r_squared = linear_regressor.score(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1), sample_weight=W[non_nan_values].ravel())
        else:
            linear_regressor.fit(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1))
            r_squared = linear_regressor.score(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1))
        Y_pred = linear_regressor.predict(X[non_nan_values].reshape(-1, 1))
        plt.figure()
        ax = combined_df_all.plot.scatter('conflicts_predicted', 'conflicts_per_active_hr')
        plt.plot(X[non_nan_values].reshape(-1, 1), Y_pred, color='red')
        plt.title("$R^2={:.4f}$".format(r_squared))
        plt.show()
