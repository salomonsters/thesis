import ast
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clustering import lat2y, lon2x
from nl_airspace_def import ehaa_airspace
CRE_fmt = "{row[timestamp]}> CRE {callsign}, {type}, {row[lat]}, {row[lon]}, {row[trk]}, {row[alt]}, {row[gs]}"
MOVE_fmt = "{row[timestamp]}> MOVE {callsign},{row[lat]}, {row[lon]},{row[alt]},{row[trk]},{row[gs]}{roc}"
DEL_fmt = "{timestamp}> DEL {callsign}"
timestamp_fmt = '%H:%M:%S.%f'
init_commands = ['    CDMETHOD STATEBASED',
                 '   ZONER 3',
                 '  CRELOG CONFLOG 1.0 Conflict log',
                 ' CONFLOG ADD FROM traf.cd confpairs dcpa tcpa tLOS qdr dist',
                 'CONFLOG ON'
                 ]
end_commands = ['CONFLOG OFF',
                'HOLD']

def bluesky_exporter(flight_data, ts_0, delete_after=10):
    positions = flight_data.sort_values(by='ts')
    positions = positions[~pd.isna(positions['trk'])]
    #TODO: possibly augment missing trk/gs/roc data by interpolating and/or calculating based on change in position?
    timestamps = pd.to_datetime(positions['ts'] - ts_0, unit='s')
    positions['timestamp'] = timestamps.dt.strftime(timestamp_fmt)
    del_timestamp = (timestamps.iloc[-1] + pd.Timedelta(10, unit='s')).strftime(timestamp_fmt)
    callsign = positions[~positions['callsign'].isna()]['callsign'].unique()[0]
    type = '_'

    def gen(positions, callsign, type, del_timestamp):
        for i, row in positions.reset_index().iterrows():
            if i == 0:
                yield CRE_fmt.format(callsign=callsign, type=type, row=row)
            else:
                if pd.isna(row['roc']):
                    roc = ''
                else:
                    roc = ',{}'.format(row['roc'])
                yield MOVE_fmt.format(callsign=callsign, row=row, roc=roc)
        yield DEL_fmt.format(callsign=callsign, timestamp=del_timestamp)
    return gen(positions, callsign, type, del_timestamp)


if __name__ == "__main__":

    scn_filename = 'scenarios/cluster_48_51.scn'
    clusters_to_analyse = [48, 51]
    n_data_points = 200
    data_date = '20180101'
    with open('data/clustered/eham_{0}.csv'.format(data_date), 'r') as fp:
        parameters = ast.literal_eval(fp.readline())
        df_all = pd.read_csv(fp)
    df_all.sort_values(by=['cluster', 'ts'], inplace=True)
    df_all.query('cluster in @clusters_to_analyse', inplace=True)
    fids = df_all['fid'].unique()
    ts_0 = df_all['ts'].min()
    zero_ts_formatted = pd.Timestamp(0).strftime(timestamp_fmt)
    final_ts_formatted = pd.Timestamp(df_all['ts'].max() - ts_0, unit='s').strftime(timestamp_fmt)
    with open(scn_filename, 'w') as f:
        for cmd in init_commands:
            f.write("{0} > {1}\n".format(zero_ts_formatted, cmd))
        for fid in fids:
            flight_data = df_all.query('fid == @fid')
            # ts_0 = flight_data['ts'].min()
            bs_gen = bluesky_exporter(flight_data, ts_0)
            for cmd in bs_gen:
                f.write(cmd + "\n")
        for cmd in end_commands:
            f.write("{0} > {1}\n".format(final_ts_formatted, cmd))
    subprocess.check_call(["sort", scn_filename, "/o", scn_filename])
    print("Done")