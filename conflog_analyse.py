import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import cache_pickle


def unique_conflicts(df, return_df_conflicts=False, dt_before_new_conflict=None, minimim_duration_for_conflict=None):
    if 'ac1' not in df.columns or 'ac2' not in df.columns:
        raise ValueError("Need ac1 and ac2 in columns")
    df = df.sort_values(['simt', 'dcpa', 'ac1']).reset_index(drop=True)
    conflicts = []
    simt_last_conflict = dict()
    distinct_conflicts_per_pair = dict()
    simt_first_conflict_occurence = dict()
    previous_ts_aircraft_pairs = ()
    this_ts_aircraft_pairs = []
    previous_simtime = 0

    for simt, df_simt in df.groupby('simt'):
        start_on_even_rows = True
        for i in range(df_simt.shape[0]):
            if (start_on_even_rows and i % 2 == 1) or (not start_on_even_rows and i % 2 == 0):
                continue
            row = df_simt.iloc[i]
            row_to_append = list(row)
            if row['ac1'] > row['ac2']:  # check if we should take the second row
                aircraft_pair = tuple(row[['ac2', 'ac1']])
                row_to_append[list(df.columns).index('ac1')] = row['ac2']
                row_to_append[list(df.columns).index('ac2')] = row['ac1']
            else:
                aircraft_pair = tuple(row[['ac1', 'ac2']])
            if i + 1 >= df_simt.shape[0] or not (row['ac1'] == df_simt.iloc[i+1]['ac2'] and row['ac2'] == df_simt.iloc[i+1]['ac1']):
                start_on_even_rows = not start_on_even_rows

            conflicts.append(row_to_append)
            if aircraft_pair in distinct_conflicts_per_pair.keys():
                if minimim_duration_for_conflict is not None and distinct_conflicts_per_pair[aircraft_pair] == 0:
                    if aircraft_pair not in previous_ts_aircraft_pairs:
                        # We want a conflict in consecutive timesteps, this is apparently not the case.
                        simt_first_conflict_occurence.pop(aircraft_pair)
                        distinct_conflicts_per_pair.pop(aircraft_pair)
                    if aircraft_pair in simt_first_conflict_occurence and simt_first_conflict_occurence[aircraft_pair] + minimim_duration_for_conflict >= simt:
                        distinct_conflicts_per_pair[aircraft_pair] = 1
                        simt_first_conflict_occurence.pop(aircraft_pair)
                else:
                    if dt_before_new_conflict is None:
                        if int(simt_last_conflict[aircraft_pair]) < int(previous_simtime):
                            distinct_conflicts_per_pair[aircraft_pair] += 1
                    else:
                        if round(simt) > round(simt_last_conflict[aircraft_pair]) + int(dt_before_new_conflict):
                            distinct_conflicts_per_pair[aircraft_pair] += 1
            else:
                if minimim_duration_for_conflict is None:
                    distinct_conflicts_per_pair[aircraft_pair] = 1
                else:
                    distinct_conflicts_per_pair[aircraft_pair] = 0
                    simt_first_conflict_occurence[aircraft_pair] = simt
            this_ts_aircraft_pairs.append(aircraft_pair)
            simt_last_conflict[aircraft_pair] = simt
        previous_simtime = simt
        previous_ts_aircraft_pairs = tuple(this_ts_aircraft_pairs)
        this_ts_aircraft_pairs = []
    if minimim_duration_for_conflict is not None:
        distinct_conflicts_per_pair = {k: v for k, v in distinct_conflicts_per_pair.items() if v > 0}
    df_conflicts = pd.DataFrame.from_records(conflicts, columns=df.columns)
    if return_df_conflicts:
        return distinct_conflicts_per_pair, df_conflicts
    return distinct_conflicts_per_pair


@cache_pickle(verbose=True)
def get_conflict_counts_from_logfile(filename, logdir = './', return_df_conflicts=False, dt_before_new_conflict=None, minimim_duration_for_conflict=None):
    with open(os.path.join(logdir, filename), 'r') as f:
        f.readline()
        columns = f.readline()[1:].replace('\n', '').replace(' ', '').replace('confpairs', 'ac1,ac2').split(",")
        df = pd.read_csv(f, names=columns)
        return unique_conflicts(df, return_df_conflicts, dt_before_new_conflict=dt_before_new_conflict, minimim_duration_for_conflict=minimim_duration_for_conflict)


def inter_and_intra_cluster_conflicts(conflict_counts):
    n_matrix = 0
    conflict_tuples = []
    for (ac1, ac2), count in conflict_counts.items():
        fid1, cluster1 = ac1.split('_')[:2]
        fid2, cluster2 = ac2.split('_')[:2]
        # Unclustered have cluster=-1 but we need semipositive indices, and cluster 0 shouldn't exist
        # but let's check that to be sure...
        if cluster1 == 0 or cluster2 == 0:
            raise ValueError("Cluster 0 should not be in use; reserved for unclustered values")
        if cluster1 == -1:
            cluster1 = 0
            print(f"Conflict for ac {ac1}")
        if cluster2 == -1:
            cluster2 = 0
            print(f"Conflict for ac {ac2}")
        # Always have cluster1 <= cluster2
        cluster1, cluster2 = int(cluster1), int(cluster2)
        if cluster1 > cluster2:
            cluster1, cluster2 = cluster2, cluster1
        n_matrix = np.max((n_matrix, cluster1+1, cluster2+1))
        conflict_tuples.append((cluster1, cluster2, count))
    cluster_matrix = np.zeros((n_matrix, n_matrix), dtype=int)
    for i, j, increment in conflict_tuples:
        cluster_matrix[i, j] = cluster_matrix[i, j] + increment
        if i != j:
            cluster_matrix[j, i] = cluster_matrix[j, i] + increment
    return cluster_matrix



if __name__ == "__main__":
    logdir = '../bluesky/output'
    lognames = [#'CONFLOG_20180101_58_83_20200421_16-20-00.log',
                # 'CONFLOG_20180101_48_20200422_14-16-20.log',
                #'CONFLOG_20180101_51_20200421_15-44-58.log',
                #'CONFLOG_20180101_48_51_20200421_15-48-14.log',
                #'CONFLOG_20180101_48_62_20200422_12-19-36.log',
                # 'CONFLOG_20180101_48_78_20200422_14-20-01.log',
                # 'CONFLOG_20180101_78_20200422_12-36-47.log',
                # 'CONFLOG_20180101_eham_airspace_20200421_14-38-56.log',
                # 'CONFLOG_val_uniform_20200522_09-29-06.log',
                'CONFLOG_val_uniform_20200522_09-42-25.log'
                ]
    records = []
    # df_own = get_conflict_counts_from_logfile(lognames[0], logdir, dt_before_new_conflict=10, minimim_duration_for_conflict=3, return_df_conflicts=True)[1]
    # df_cross = get_conflict_counts_from_logfile(lognames[1], logdir, dt_before_new_conflict=10, minimim_duration_for_conflict=3, return_df_conflicts=True)[1]
    for logname in lognames:
        conflict_counts = get_conflict_counts_from_logfile(logname, logdir, dt_before_new_conflict=10, minimim_duration_for_conflict=3)
        conflicts_by_cluster = inter_and_intra_cluster_conflicts(conflict_counts)
        # if logname != lognames[-1]: # check the first entries against the full package
        print(logname)
        for idx in np.argwhere(conflicts_by_cluster):
            print(idx, conflicts_by_cluster[idx[0], idx[1]])
        f, ax = plt.subplots(figsize=(22, 18))
        nz = np.unique(conflicts_by_cluster.nonzero())
        mask = np.triu(np.ones_like(conflicts_by_cluster[nz,:][:, nz], dtype=np.bool), k=1) | (conflicts_by_cluster[nz,:][:, nz] == 0)
        cmap = sns.light_palette("green", as_cmap=True)
        sns.heatmap(pd.DataFrame(conflicts_by_cluster[nz,:][:, nz], columns=nz, index=nz), mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt="d")
        plt.show()
        np.savetxt('data/scenario_analysis/{0}'.format(logname.replace('.log', '_analysis.csv')), conflicts_by_cluster,
                   fmt='%d', delimiter=',')
        # records.append({"-".join(k): v for k, v in conflict_counts.items()})
        # print(f"{logname = }\n\t{conflict_counts = }")
    # df_all_conflicts = pd.DataFrame.from_records(records, index=lognames).T

