import numpy as np
import pandas as pd

in_filename_format = 'data/adsb_decoded_in_eham/ADSB_DECODED_{0}.csv.gz'
out_filename_format = 'data/adsb_decoded_in_eham_combined/{0}_split_{1}.csv.gz'
data_dates = ['20180101', '20180102', '20180104', '20180105']
split_in = 4
seed = 42
# seed = None
rng = np.random.RandomState(seed)
df_list = [[] for _ in range(split_in)]
# First populate df_list
for date_i, data_date in enumerate(data_dates):
    df = pd.read_csv(in_filename_format.format(data_date))
    unique_fids = df['fid'].unique()
    N = unique_fids.shape[0]
    chunk_size = N / split_in
    rng.shuffle(unique_fids)

    for i in range(split_in):
        fids_split = unique_fids[int(chunk_size * i):int(chunk_size * (i + 1))]
        print("Using {} fids from {}".format(fids_split.shape[0], data_date))
        df_list[i].append(df.query("fid in @fids_split"))
# Then walk through df_list, use concat and save
for i in range(split_in):
    df_out = pd.concat(df_list[i])
    df_out['ts_original'] = df_out['ts'].copy(deep=True)
    # Create new ts column which has time in seconds but as if everything happened on one day
    df_out['t'] = pd.TimedeltaIndex(df_out['ts'], unit='s')
    df_out['t'] = df_out['t'] - df_out['t'].dt.floor('D')
    df_out['ts'] = df_out['t'].dt.total_seconds()
    df_out.drop(columns=['t'])
    out_filename = out_filename_format.format("-".join(data_dates), i)
    df_out.to_csv(out_filename)
    print("Saved to {}".format(out_filename))
