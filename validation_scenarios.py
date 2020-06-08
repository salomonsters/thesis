import numpy as np
from numpy.random import default_rng
import pandas as pd
import bluesky_export
from nl_airspace_def import eham_ctr, eham_ctr_centre
from nl_airspace_helpers import parse_lat_lon
import geopandas


def generate_start_positions(target, radius, headings):
    bearings = np.pi - headings
    x_start = target[0] - np.degrees(radius / earth_r * np.sin(bearings) / np.cos(target[1]))
    y_start = target[1] - np.degrees(radius / earth_r * np.cos(bearings))
    return x_start, y_start

earth_r = 6371000  # m
target_x, target_y = parse_lat_lon(eham_ctr_centre)

if __name__ == "__main__":
    seed = 1337
    rg = default_rng(seed)
    scenario_filename = 'scenarios/velocity_U60-80_U80-100_U-100-120_heading_C0_90_spawn_U400-600.scn'
    n_aircraft = 40
    # spawn_interval = lambda: rg.exponential(60, n_aircraft)
    spawn_interval = lambda: rg.uniform(400, 600, n_aircraft*2)[n_aircraft:]
    # spawn_interval = lambda: 60*np.ones(n_aircraft)
    radius = 15000  # m
    V_aircraft = [rg.uniform(60, 80, n_aircraft), rg.uniform(80, 100, n_aircraft), rg.uniform(100, 120, n_aircraft)]
    h_aircraft = 2000  # ft
    headings = 0, 90
    time_resolution = 5  # s

    df_elements = []
    cluster_n = 0
    for speeds in V_aircraft:
        for heading in headings:
            spawn_times = np.round(np.cumsum(np.hstack([0, spawn_interval()[:-1]])))

            x_start, y_start = generate_start_positions((target_x, target_y), radius, np.radians(180 + np.array(heading)))

            for i in range(n_aircraft):
                speed = speeds[i]
                n_points = int(radius / (speed * time_resolution)) + 1

                df_elements.append(pd.DataFrame.from_dict({
                'callsign': n_points * ["{1:03d}_{0:02d}_{2:03d}".format(cluster_n, i + 1, int(speed))],
                'fid':  n_points * ["{1:03d}_{0:02d}_{2:03d}".format(cluster_n, i + 1, int(speed))],
                'lon': np.linspace(x_start, target_x, n_points),
                'lat': np.linspace(y_start, target_y, n_points),
                'trk': n_points * [heading],
                'gs':  n_points * [speed],
                'alt':  n_points * [h_aircraft],
                'roc':  n_points * [0],
                    'ts': np.linspace(spawn_times[i], spawn_times[i] + radius/speed, n_points),
                    'action':  ['create'] + (n_points - 2) * ['move'] + ['delete']
                }))
            cluster_n += 1
    df = pd.concat(df_elements)
    timestamps = pd.to_datetime(df['ts'], unit='s')
    df['timestamp'] = timestamps.dt.strftime(bluesky_export.timestamp_fmt)
    df.sort_values(by=['ts', 'callsign'], inplace=True)
    zero_ts_formatted = pd.Timestamp(0).strftime(bluesky_export.timestamp_fmt)
    final_ts_formatted = pd.Timestamp(df['ts'].max(), unit='s').strftime(bluesky_export.timestamp_fmt)

    with open(scenario_filename, 'w') as f:
        for cmd in bluesky_export.init_commands:
            f.write("{0} > {1}\n".format(zero_ts_formatted, cmd))
        for _, row in df.iterrows():
            if row['action'] == 'create':
                line = bluesky_export.CRE_fmt.format(callsign=row['callsign'], type='_', row=row)
            elif row['action'] == 'move':
                line = bluesky_export.MOVE_fmt.format(callsign=row['callsign'], row=row, roc='')
            elif row['action'] == 'delete':
                line = bluesky_export.DEL_fmt.format(callsign=row['callsign'], timestamp=row['timestamp'])
            else:
                raise ValueError("Invalid action encountered in row {}".format(row))
            f.write(line + "\n")
        for cmd in bluesky_export.end_commands:
            f.write("{0} > {1}\n".format(final_ts_formatted, cmd))
    print("Done")




