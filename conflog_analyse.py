import os
import pprint

import numpy as np
import pandas as pd


def unique_conflicts(df, return_df_conflicts=False):
    if 'ac1' not in df.columns or 'ac2' not in df.columns:
        raise ValueError("Need ac1 and ac2 in columns")
    conflicts = []
    distinct_conflicts_per_pair = dict()
    previous_ts_aircraft_pairs = []
    previous_simtime = 0
    for i in range(0, df.shape[0], 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i+1]
        if not (row1['ac1'] == row2['ac2'] and row1['ac2'] == row2['ac1']):
            raise ValueError("Conflog should contain the same aircraft pair in pairs of two rows")
        take_second_row = row1['ac1'] > row1['ac2'] # check if we should take the second row
        aircraft_pair = tuple(df.iloc[i + int(take_second_row)][['ac1', 'ac2']])
        conflicts.append(df.iloc[i + int(take_second_row)])
        if aircraft_pair in distinct_conflicts_per_pair.keys():
            if aircraft_pair not in previous_ts_aircraft_pairs:
                distinct_conflicts_per_pair[aircraft_pair] += 1
        else:
            distinct_conflicts_per_pair[aircraft_pair] = 1
        if row1['simt'] > previous_simtime:
            previous_ts_aircraft_pairs = [aircraft_pair]
        else:
            previous_ts_aircraft_pairs.append(aircraft_pair)
        previous_simtime = row1['simt']

    df_conflicts = pd.DataFrame.from_records(conflicts, columns=df.columns)
    if return_df_conflicts:
        return distinct_conflicts_per_pair, df_conflicts
    return distinct_conflicts_per_pair


def get_conflict_counts_from_logfile(filename, logdir = './', return_df_conflicts=False):
    with open(os.path.join(logdir, filename), 'r') as f:
        f.readline()
        columns = f.readline()[1:].replace('\n', '').replace(' ', '').replace('confpairs', 'ac1,ac2').split(",")
        df = pd.read_csv(f, names=columns)
        return unique_conflicts(df, return_df_conflicts)


if __name__ == "__main__":
    logdir = '../bluesky/output'
    lognames = ['CONFLOG_cluster_48_20200421_11-47-22.log', 'CONFLOG_cluster_51_20200421_11-51-08.log', 'CONFLOG_cluster_48_51_20200421_11-54-20.log']

    for logname in lognames:
        conflict_counts = get_conflict_counts_from_logfile(logname, logdir)
        print(f"{logname = }\n\t{conflict_counts = }")

