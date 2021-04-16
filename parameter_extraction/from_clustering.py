import ast
import warnings

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from scipy.stats import circmean

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
        return ('all_av', None)
    elif where_array[0] == 0 and where_array.shape[0] > 2:# and where_array[2] == n_data_points: # where_array[1] > n_data_points*0.15:
        return ('diverging_av', None)
    first_loss_of_separation_index = where_array[0]
    assert first_loss_of_separation_index != 0
    return ('converging_av', (first_loss_of_separation_index, first_loss_of_separation_index))

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

def index_of_closest_point(row):
    v_diff = row['mean_alt_i'] - row['mean_alt_j']
    within_S_v = np.abs(v_diff <= S_v)
    assert np.any(within_S_v), "index_of_closest_points should be called on a row where there is overlap"

    h_distances = np.linalg.norm(row['mean_track_i'] - row['mean_track_j'], axis=1)
    h_distances_masked = np.ma.masked_array(h_distances, mask=~within_S_v)
    return h_distances_masked.argmin()


def overlap(left, right):
    if left['cluster'] == right['cluster']:
        if left['cluster'] == 0:
            return ('unclustered-unclustered', None)
        return ('within-cluster', None)
    elif left['cluster'] == 0 or right['cluster'] == 0:
        return ('clustered-unclustered', None)
    v_diff = left.mean_alt - right.mean_alt
    h_distances = np.linalg.norm(left.mean_track - right.mean_track, axis=1)
    within_S_h_and_S_v_inner = (np.abs(v_diff) <= 1*S_v) & (h_distances <= 1*S_h)

    within_S_h_and_S_v_outer = (np.linalg.norm(left.mean_track[:, None] - right.mean_track[None, :], axis=2) < 1*S_h) & (
                np.abs(left.mean_alt[:, None] - right.mean_alt[None, :]) <= 1*S_v)
    left_track_predicted = (left.mean_track + (left['other_states'][:, other_states_gs_index] * [np.sin(np.radians(left['other_states'][:, other_states_trk_index])), np.cos(np.radians(left['other_states'][:, other_states_trk_index]))]  * t_l*60).T)
    right_track_predicted = (right.mean_track + (right['other_states'][:, other_states_gs_index] * [np.sin(np.radians(right['other_states'][:, other_states_trk_index])), np.cos(np.radians(right['other_states'][:, other_states_trk_index]))]  * t_l*60).T)

    within_S_h_and_predicted_within_S_v = (np.linalg.norm(left_track_predicted[:, None] - right_track_predicted[None, :], axis=2) < 1*S_h) &\
                                          ((left.mean_alt + left['other_states'][:, 2] * t_l*60)[:, None] - (right.mean_alt + right['other_states'][:, 2] * t_l*60)[None, :] <= 1 * S_v)
    if np.any(within_S_h_and_S_v_inner):
        return overlap_type(within_S_h_and_S_v_inner)
    elif np.any(within_S_h_and_S_v_outer):
        return ('cross', (int(np.median(np.unique(np.argwhere(within_S_h_and_S_v_outer)[:, 0]))),
                int(np.median(np.unique(np.argwhere(within_S_h_and_S_v_outer)[:, 1])))))
    elif np.any(within_S_h_and_predicted_within_S_v):
        if np.any(close := np.array([np.abs(a-b)<5 for a, b in np.argwhere(within_S_h_and_predicted_within_S_v)])):
            return ('converging_la', np.argwhere(within_S_h_and_predicted_within_S_v)[close].min(axis=0))
        else:
            return ('cross_la', (int(np.median(np.unique(np.argwhere(within_S_h_and_predicted_within_S_v)[:, 0]))),
                int(np.median(np.unique(np.argwhere(within_S_h_and_predicted_within_S_v)[:, 1])))))
    else:
        return ('none', None)


    # consecutive_lengths = consecutive_string_lengths(within_S_h & within_S_v)

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
    at_i, at_j = np.array(at).astype(int)
    gs_i = row['other_states_i'][at_i, other_states_gs_index]
    gs_j = row['other_states_j'][at_j, other_states_gs_index]
    trk_i = row['other_states_i'][at_i, other_states_trk_index]
    trk_j = row['other_states_j'][at_j, other_states_trk_index]
    trk_diff = np.radians(np.abs(trk_i-trk_j))
    mean_time_between_activations_i = row['mean_time_between_activations_i']
    mean_time_between_activations_j = row['mean_time_between_activations_j']
    flights_as_events_lambda_correction = 3600*3600/(mean_time_between_activations_i*mean_time_between_activations_j)
    if trk_diff > np.pi:
        trk_diff = trk_diff - np.pi
    V_rel = np.sqrt(gs_i**2+gs_j**2-2*gs_i*gs_j*np.cos(trk_diff))
    hourly_arrival_rates_prod = row['hourly_arrival_rates_prod']# * row['hourly_arrival_rates_j']
    nonzero_hourly_arrival_rates_prod = hourly_arrival_rates_prod[hourly_arrival_rates_prod > 0]
    if len(nonzero_hourly_arrival_rates_prod) == 0:
        lam_correction = 0
    else:
        lam_correction = nonzero_hourly_arrival_rates_prod.max()
    V_rel_corr = V_rel /np.sin(trk_diff)/(gs_i*gs_j) * lam_correction
    return V_rel, V_rel_corr, trk_diff, gs_i, gs_j, hourly_arrival_rates_prod, flights_as_events_lambda_correction


from sklearn.linear_model import LinearRegression


def find_and_plot_correlation(df, x_col, y_col, ax, selector=None, trendline=True, scatter_kwargs={}, trend_kwargs={}):
    # combined_df_all.query('conflicts_predicted<20', inplace=True)

    X = df[x_col].values.reshape(-1, 1)
    Y = df[y_col].values.reshape(-1, 1)
    W = df['track_product'].values.reshape(-1, 1)
    # if x_col == 'conflicts_predicted_2':
    #     non_nan_values = ~pd.isna(X + Y) & np.isfinite(X + Y) & (Y < 4) & (X < 0.5)
    # else:
    # non_nan_values = ~pd.isna(X+Y) & np.isfinite(X+Y) & (Y<4) & (X<0.5) & (Y>0.01)
    non_nan_values = ~pd.isna(X + Y) & np.isfinite(X + Y)
    if selector:
        non_nan_values = non_nan_values & selector(X,Y)
    linear_regressor = LinearRegression()
    if use_weighted_least_squares:
        linear_regressor.fit(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1),
                             sample_weight=W[non_nan_values].ravel())
        r_squared = linear_regressor.score(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1),
                                           sample_weight=W[non_nan_values].ravel())
    else:
        linear_regressor.fit(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1))
        r_squared = linear_regressor.score(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1))
    Y_pred = linear_regressor.predict(X[non_nan_values].reshape(-1, 1))
    # plt.figure()
    df.loc[non_nan_values].plot.scatter(x_col, y_col, ax=ax, **scatter_kwargs)
    if trendline:
        ax.plot(X[non_nan_values].reshape(-1, 1), Y_pred, color='red', linestyle='--', **trend_kwargs)
    return "$R^2={:.4f}$".format(r_squared)

if __name__ == "__main__":
    S_h_in_nm = 3
    S_h = 1852 * S_h_in_nm
    S_v = 1000
    t_l = 5./60.
    conflicts_x_and_y_lim_for_plot = 10
    timeshift = None
    as_events = False
    use_weighted_least_squares = False
    calculate_V_rel_method = 'first'  # 'closest' or 'first'

    timeshift_suffix = ''
    filename_prefix = 'eham_stop_mean_0.25_std_0.25'
    t_col = 'ts'
    if timeshift is not None:
        timeshift_suffix = '-timeshift-uniform-0-{}'.format(timeshift)
    elif as_events:
        timeshift_suffix = '-as-events-repeats-5'
        conflicts_x_and_y_lim_for_plot = 80

    replay_results_file = 'data/conflict_replay_results/{}_2018010[1-2-4-5]-S_h-{}-S_v-{}-t_l-{:.4f}{}.xlsx'.format(filename_prefix, S_h_in_nm, S_v, t_l, timeshift_suffix)
    replay_df_all_splits = pd.read_excel(replay_results_file)
    combined_df_list = []
    for split in [1,2,4,5]:
        data_date = '2018010{}'.format(split)
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

        with open('data/clustered/{2}_{0}{1}.csv'.format(data_date, timeshift_suffix, filename_prefix), 'r') as fp:
            parameters = None
            if not as_events:
                parameters = ast.literal_eval(fp.readline())
            df = pd.read_csv(fp)
        df.sort_values(by=['cluster', t_col], inplace=True)
        df[t_col] = df[t_col] - df[t_col].min()
        tracks_per_cluster = df.groupby('cluster')['fid'].unique().apply(len)
        rows = []
        # for cluster, n_tracks in tracks_per_cluster.sort_values(ascending=False).iteritems():
        df_tracks_per_cluster = pd.DataFrame(tracks_per_cluster)
        print("Split, total fid's, n_clusters, n_tracks mean pm std, unclustered : {0} & {4} & {3} & {2[0]:.2f} $\pm$ {2[1]:.2f} & {1} ({5:.2f}\%)".format(
            split+1, tracks_per_cluster.iloc[0],
            df_tracks_per_cluster.query('cluster>0').agg(['mean', 'std']).to_numpy().ravel(),
            len(df_tracks_per_cluster.index) - 1, len(df['fid'].unique()), 100*tracks_per_cluster.iloc[0]/len(df['fid'].unique())))
        assert len(tracks_per_cluster) > 2
        for cluster, n_tracks in tracks_per_cluster.iteritems():

            cluster_points = df.query('cluster== @cluster')
            if cluster == -1:
                cluster = 0
                cluster_points['cluster'] = 0
            cluster_points = cluster_points.assign(index_along_track=-1)
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
                # # Stupid SettingWithCopyWarning
                # assert np.all(cluster_points.loc[track_points.index[resampling_indices], 'index_along_track'] >= 0)
                cluster_ts_0_items.append(track_points[t_col].min())
            points = cluster_points.query('index_along_track >= 0')
            points_per_fid = points.set_index(['fid', 'index_along_track']).sort_index()[columns].to_numpy().reshape(n_tracks, 200, len(columns))
            # plt.figure()
            # medoid_index = medoid(points_per_fid.reshape(n_tracks, -1), axis=0, indexonly=True)
            mean = np.nanmean(points_per_fid.reshape(n_tracks, -1), axis=0).reshape(n_data_points, len(columns))
            if 'trk' in columns:
                trk_mean = circmean(points_per_fid[:,:, columns.index('trk')], high=360, axis=0, nan_policy='omit')
                mean[:, columns.index('trk')] = trk_mean
            mean_2d_track = mean[:, :2]
            mean_alt = mean[:, 2]
            if mean.shape[1] > 2:
                other_states = mean[:, 3:]
            else:
                other_states = None
            mean_time_between_activations = np.diff(np.sort(cluster_ts_0_items))[1:].mean()
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

            rows.append((cluster, n_tracks, LineString(mean_2d_track), mean_2d_track, mean_alt, other_states, active_duration_in_hours, t_start, t_end, hourly_arrival_rates, mean_time_between_activations))

            # plt.figure()
            # points.plot.scatter(y='index_along_track', x='gs', s=0.01)
            # plt.show()

        gdf = geopandas.GeoDataFrame.from_records(rows, columns=['cluster', 'n_tracks', 'geometry', 'mean_track', 'mean_alt', 'other_states', 'active_duration_in_hours', 't_start', 't_end','hourly_arrival_rates', 'mean_time_between_activations'])
        # gdf['hourly_arrivals'] = gdf['n_tracks']/gdf['active_duration_in_hours']
        # intersection_map = np.array([[left.intersects(right) for left in gdf['geometry']] for right in gdf['geometry']])
        intersection_heatmap = np.array([[overlap(left, right) for _, left in gdf.iterrows()] for _, right in gdf.iterrows()], dtype='object')
        intersection_heatmap_extra_row_and_col = np.zeros(shape=np.array(intersection_heatmap.shape), dtype='object')
        intersection_heatmap_extra_row_and_col[:, :] = np.nan
        intersection_heatmap_extra_row_and_col[0:, 0:] = intersection_heatmap
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
        overlap_tuples = combined_df.apply(lambda row: intersection_heatmap_extra_row_and_col[row['i'], row['j']], axis=1).apply(pd.Series)
        overlap_tuples.columns=['overlap_type', 'first_overlap_index']
        combined_df = combined_df.merge(overlap_tuples, left_index=True, right_index=True)
        # combined_df.query('i>0 and j>0', inplace=True)
        combined_df['track_product'] = combined_df['n_tracks_i'] * combined_df['n_tracks_j']
        combined_df['combined_active_hrs'] = combined_df.apply(lambda x: np.max(x[['t_end_i', 't_end_j']]) -  np.min(x[['t_start_i', 't_start_j']]), axis=1)/3600
        combined_df['conflicts_per_active_hr'] = combined_df['conflicts'] / combined_df['combined_active_hrs']


        # combined_df.plot.hexbin(x='hourly_arrivals_i', y='hourly_arrivals_j', C='conflicts')
        # plt.show()

        # Todo tue: use high-overlap clusters to find method to find location/properties of intersection
        # Perhaps: identify consecutive points and use middle? And find non-intersecting points?

        combined_df['V_rel_corr'] = np.nan
        combined_df = combined_df.assign(hourly_arrival_rates_prod=None)
        #combined_df['hourly_arrival_rates_prod'] = combined_df['hourly_arrival_rates_prod'].astype(object)
        combined_df['hourly_arrival_rates_prod'] = combined_df.apply(
            lambda row: row['hourly_arrival_rates_i'] * row['hourly_arrival_rates_j'], axis=1)
        for i in combined_df.index:
            # ovl_type = combined_df.loc[i]['overlap_type']
            # if isinstance(ovl_type, np.ndarray):
            #     shape = ovl_type.shape[0]
            #     if shape <= 2:
            if not pd.isna(np.any(at := combined_df.loc[i]['first_overlap_index'])):
                    if calculate_V_rel_method == 'closest':
                        at = index_of_closest_point(combined_df.loc[i])
                        at = (at, at)
                    elif calculate_V_rel_method == 'first':
                        pass
                    else:
                        raise ValueError("calculate_V_rel_method should be 'closest' or 'first'")
                    V_rel_corr_result = calculate_V_rel_corrected_at(combined_df.loc[i], at)
                    for k, col in enumerate(['V_rel', 'V_rel_corr', 'trk_diff', 'gs_i', 'gs_j', 'hourly_arrival_rates_prod', 'flights_as_events_lambda_correction']):
                        combined_df.at[i, col] = V_rel_corr_result[k]
        #         else:
        #             pass#print(ovl_type)
        #     pass
        combined_df_list.append(combined_df)
    combined_df_all = pd.concat(combined_df_list).reset_index()

    intensity_parameters = combined_df_all['hourly_arrival_rates_prod'].dropna().apply(
        lambda x: (np.mean(x), np.max(x), np.sum(x), np.count_nonzero(x))).apply(pd.Series)
    intensity_parameters.columns = ['mean_intensity', 'max_intensity', 'total_intensity', 'intensity_active_hours']
    combined_df_all = combined_df_all.merge(intensity_parameters, left_index=True, right_index=True, how='left')
    combined_df_all['conflicts_predicted'] = 2 * S_h / 1852 * combined_df_all['V_rel_corr']
    trk_diff = combined_df_all['trk_diff']
    gs_i = combined_df_all['gs_i']
    gs_j = combined_df_all['gs_j']
    combined_df_all['conflicts_predicted_2'] = 2 * S_h / 1852 * combined_df_all['V_rel']/np.sin(trk_diff)/(gs_i*gs_j)*combined_df_all['mean_intensity']
    combined_df_all['conflicts_predicted_3'] = 2 * S_h / 1852 * combined_df_all['V_rel']/np.sin(trk_diff)/(gs_i*gs_j)*combined_df_all['max_intensity']
    combined_df_all['conflicts_predicted_4'] = 2 * S_h / 1852 * combined_df_all['V_rel']/np.sin(trk_diff)/(gs_i*gs_j)*combined_df_all['total_intensity']
    combined_df_all['conflicts_predicted_5'] = 2 * S_h / 1852 * combined_df_all['V_rel']/np.sin(trk_diff)/(gs_i*gs_j)*combined_df_all['flights_as_events_lambda_correction']
    # if as_events:
    #     combined_df_all['conflicts_per_active_hr_based_on_intensity'] = combined_df_all['conflicts']/combined_df_all[['active_duration_in_hours_i', 'active_duration_in_hours_j']].min(axis=1)
    # else:
    combined_df_all['conflicts_per_active_hr_based_on_intensity'] = combined_df_all['conflicts'] / combined_df_all['intensity_active_hours']


    # high_conflicts = combined_df.query('type=="between"').sort_values('conflicts_per_active_hr')[-100:]


    fig, ax = plt.subplots(2, 1)
    ax0_twin = ax[0].twiny()
    combined_df_all.groupby('overlap_type')['conflicts'].agg(['sum']).plot.barh(ax=ax[0], position=0, color='C0', width=0.4)
    combined_df_all.groupby('overlap_type')['conflicts'].agg(['count']).plot.barh(ax=ax0_twin, position=1, color='C1', width=0.4)
    ax[0].set_xlabel("Number of conflicts")
    ax[0].set_ylabel("Overlap type")

    ax0_twin.set_ylabel("")
    ax0_twin.set_xlabel("Number of flow pairs")
    lines, labels = ax[0].get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    ax[0].legend(lines + lines2, ['Conflicts', 'Flow pairs'], loc='lower right')
    ax0_twin.get_legend().remove()
    ax[0].set_title('')
    y_lims = ax[0].get_ylim()
    ax[0].set_ylim((y_lims[0], y_lims[1]+0.5))
    x_lims = np.array([ax[0].get_xlim(), ax0_twin.get_xlim()]).max(axis=0)
    ax[0].set_xlim(x_lims)
    ax0_twin.set_xlim(x_lims)
    combined_df_all.boxplot(column='conflicts_per_active_hr_based_on_intensity', by='overlap_type', vert=False, ax=ax[1])
    # plt.subplots_adjust(left=0.2)
    ax[1].set_xlabel("Conflict rate [1/h]")
    ax[1].set_title('')
    ax[1].set_ylabel("Overlap type")
    fig.suptitle('')
    plt.tight_layout()
    for ext in ['png', 'eps']:
        for overleaf_project_name in ['thesis_full', 'thesis_article']:
            fig.savefig(r"C:/Users/salom/Dropbox/Apps/Overleaf/{}/figures/replay/".format(overleaf_project_name) +
                        '{}_20180101-20180102-20180104-20180105-splits_[0-1-2-3]-S_h-{}-S_v-{}-t_l-{:.4f}{}.{}'.format(filename_prefix, S_h_in_nm, S_v, t_l, timeshift_suffix, ext))
    plt.show()

    overlap_plot_i = 0
    fig, axs = plt.subplots(1, 2, figsize=(6, 5))
    markers = ('+', '.', '1', '*', 'x')
    colors = ('C0', 'C1', 'C2', 'C3', 'C4', 'C5')
    x_col = 'conflicts_predicted'
    y_col = 'conflicts_per_active_hr_based_on_intensity'
    combined_df_all['Simulated/analytical'] = combined_df_all[y_col]/combined_df_all[x_col]
    legends = []
    for overlap_type_name, overlap_group in combined_df_all.groupby('overlap_type'):
        if overlap_group['first_overlap_index'].dropna().shape[0] == 0:
            continue

        find_and_plot_correlation(overlap_group, x_col, y_col, axs[0], lambda X,Y: (X< conflicts_x_and_y_lim_for_plot) & (Y<conflicts_x_and_y_lim_for_plot) #& (0.01 <Y)
                                  , scatter_kwargs={'marker': markers[overlap_plot_i], 'color': colors[overlap_plot_i]}, trendline=False)
        legends.append(overlap_type_name)
        overlap_plot_i += 1
    axs[0].set_xlabel("Analytical conflict rate [1/hr]")
    axs[0].set_ylabel("Simulated conflict rate [1/hr]")
    axs[0].plot([0, conflicts_x_and_y_lim_for_plot], [0, conflicts_x_and_y_lim_for_plot], color=colors[overlap_plot_i], linestyle='--')
    axs[0].legend(['Theoretical'] + legends )

    combined_df_all.query("overlap_type in @legends").boxplot(column='Simulated/analytical', by='overlap_type', ax=axs[1], rot=45)
    axs[1].set_ylim([0, 3])
    axs[1].set_title('Ratio per overlap type')
    axs[1].set_xlabel("")
    axs[1].set_ylabel("Simulated/Analytical")
    axs[1].axhline(y=1)

    fig.suptitle('')
    plt.tight_layout()
    for ext in ['png', 'eps']:
        for overleaf_project_name in ['thesis_full', 'thesis_article']:
            fig.savefig(r"C:/Users/salom/Dropbox/Apps/Overleaf/{}/figures/replay/".format(overleaf_project_name) +
                'trend-{}_20180101-20180102-20180104-20180105-splits_[0-1-2-3]-S_h-{}-S_v-{}-t_l-{:.4f}{}.{}'.format(filename_prefix, S_h_in_nm, S_v, t_l, timeshift_suffix, ext))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(3,5))
    x_col = 'conflicts_predicted'
    y_col = 'conflicts_per_active_hr_based_on_intensity'
    r2 = find_and_plot_correlation(combined_df_all.query('overlap_type=="converging_av"'), x_col, y_col, ax, lambda X,Y: (X< conflicts_x_and_y_lim_for_plot) & (Y<conflicts_x_and_y_lim_for_plot))#& (0.01 <Y))
    ax.set_xlabel("Predicted conflict rate [1/hr]")
    ax.set_ylabel("Observed conflict rate [1/hr]")
    ax.legend(["Observations", 'Trendline ' + r2 ][::-1])
    fig.suptitle('Converging_av')
    plt.tight_layout()
    for ext in ['png', 'eps']:
        for overleaf_project_name in ['thesis_full', 'thesis_article']:
            fig.savefig(r"C:/Users/salom/Dropbox/Apps/Overleaf/{}/figures/replay/".format(overleaf_project_name) +
                        'trend-converging-{}_20180101-20180102-20180104-20180105-splits_[0-1-2-3]-S_h-{}-S_v-{}-t_l-{:.4f}{}.{}'.format(filename_prefix, S_h_in_nm, S_v, t_l, timeshift_suffix, ext))
    plt.show()




    if True:
        x_cols = ['conflicts_predicted', 'conflicts_predicted_2', 'conflicts_predicted_3', 'conflicts_predicted_4', 'conflicts_predicted_5']
        y_cols = ['conflicts_per_active_hr', 'conflicts_per_active_hr_based_on_intensity']
        fig, axs = plt.subplots(len(y_cols), len(x_cols), figsize=(20, 10))
        for i, y_col in enumerate(y_cols):
            for j, x_col in enumerate(x_cols):
                find_and_plot_correlation(combined_df_all, x_col, y_col, axs[i, j], lambda X,Y: (X < conflicts_x_and_y_lim_for_plot) & (Y < conflicts_x_and_y_lim_for_plot))
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(replay_results_file[replay_results_file.rfind('/'):])
        plt.show()
                # plt.figure()
                # combined_df_all.plot.scatter('trk_diff', 'conflicts_per_active_hr')
                # plt.show()

    combined_df_all['has_conflicts'] = combined_df_all['conflicts'].astype(bool)
    print(combined_df_all.groupby(['overlap_type', 'has_conflicts']).size().unstack().assign(
        sum=lambda x: np.nansum(x, axis=1))[['sum', True, False]].rename(
        columns={"sum": "Total No. of flow pairs", True: "With conflicts", False: "Without conflicts"}
    ).to_latex(float_format="{:g}".format))

