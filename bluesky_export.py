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
                 'CONFLOG ON',
                 'FF'
                 ]
end_commands = ['CONFLOG OFF',
                'HOLD']

def bluesky_exporter(flight_data, ts_0, delete_after=10, callsign_column='callsign'):
    positions = flight_data.sort_values(by='ts')
    positions = positions[~pd.isna(positions['trk'])]
    #TODO: possibly augment missing trk/gs/roc data by interpolating and/or calculating based on change in position?
    timestamps = pd.to_datetime(positions['ts'] - ts_0, unit='s')
    positions['timestamp'] = timestamps.dt.strftime(timestamp_fmt)
    del_timestamp = (timestamps.iloc[-1] + pd.Timedelta(delete_after, unit='s')).strftime(timestamp_fmt)
    callsign = positions[~positions[callsign_column].isna()][callsign_column].unique()[0]
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


def export_to_file(df, filename, ts_0=0, callsign_column='callsign'):
    zero_ts_formatted = pd.Timestamp(0).strftime(timestamp_fmt)
    final_ts_formatted = pd.Timestamp(df['ts'].max() - ts_0, unit='s').strftime(timestamp_fmt)
    fids = df['fid'].unique()
    with open(filename, 'w') as f:
        for cmd in init_commands:
            f.write("{0} > {1}\n".format(zero_ts_formatted, cmd))
        for fid in fids:
            flight_data = df.query('fid == @fid')
            # ts_0 = flight_data['ts'].min()
            bs_gen = bluesky_exporter(flight_data, ts_0, callsign_column=callsign_column)
            for cmd in bs_gen:
                f.write(cmd + "\n")
        for cmd in end_commands:
            f.write("{0} > {1}\n".format(final_ts_formatted, cmd))

if __name__ == "__main__":

    data_date = '20180101'
    clusters_to_analyse = [48, 78]
    scn_filename = f'scenarios/{data_date}_48_78.scn'
    with open('data/clustered/eham_{0}.csv'.format(data_date), 'r') as fp:
        parameters = ast.literal_eval(fp.readline())
        df_all = pd.read_csv(fp)
    ts_0 = df_all['ts'].min()
    df_all.sort_values(by=['cluster', 'ts'], inplace=True)

    df_all.query('cluster in @clusters_to_analyse', inplace=True)
    df_all['fid_cluster'] = df_all.apply(lambda r: "{fid}_{cluster}".format(**r), axis=1)
    export_to_file(df_all, scn_filename, ts_0, 'fid_cluster')

    subprocess.check_call(["sort", scn_filename, "/o", scn_filename])
    print("Done")
