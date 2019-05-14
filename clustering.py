import numpy as np
import pandas as pd
import itertools
import scipy as sp
import geopandas
from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import glob
import os

# airspace_query = "airport=='EHRD' and type=='CTR'"
# zoom = 15
#
# airspace = ehaa_airspace.query(airspace_query)

n_data_points = 1000
sigma = 1

## SINGLE FILENAME
df = pd.read_csv('data/adsb_decoded_in_ehrd/ADSB_DECODED_20180101.csv.gz')#, converters={'callsign': lambda s: s.replace('_', '')})
## ENTIRE DIRECTORY
# path = './data/adsb_decoded_in_ehrd/'
# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))))

# TODO: only use segments above ground, that way the similarity is really on route, not ground segment
# TODO: then maybe pad as well so we don't get nan's?

def adjacency_matrix(x_i, x_j):
    x = x_i - x_j
    x.dropna(inplace=True)
    if len(x) == 0:
        return 0
    dist = np.linalg.norm(x)
    return np.exp(-dist ** 2 / (2. * (sigma * sigma)))

def scale_and_average_df(df, fields=['lon', 'lat']):
    ts_min, ts_max = df['ts'].min(), df['ts'].max()
    ts_len = ts_max - ts_min
    df['ts_scaled'] = (df['ts'] - ts_min)/ts_len*n_data_points
    df['timestamp'] = df['ts_scaled'].apply(lambda x: pd.Timestamp(x, unit='s'))
    df.set_index('timestamp', inplace=True)
    resampled = df.resample(rule='1s')[fields].mean()
    return resampled


df_g = df.groupby('fid')
num_groups = len(df_g)

W = np.zeros((num_groups, num_groups), dtype=float)
i = 0

processed_dfs = dict()
for fid_i, df_i in df_g:
    if fid_i in processed_dfs:
        x_i = processed_dfs[fid_i]
    else:
        x_i = scale_and_average_df(df_i)
        processed_dfs[fid_i] = x_i

    j = 0
    for fid_j, df_j in df.groupby('fid'):
        if fid_j in processed_dfs:
            x_j = processed_dfs[fid_j]
        else:
            x_j = scale_and_average_df(df_j)
            processed_dfs[fid_j] = x_j
        Wij = adjacency_matrix(x_i, x_j)
        W[i, j] = Wij
        j += 1
    i += 1
print(W)

def spectralCluster(W, omegaa_min):
    D = np.zeros_like(W)
    for i in range(W.shape[0]):
        D[i,i] = np.sum(W[i,:])
    L = D - W
    # TODO: implement second smallest eigenvector problem
    # TODO: ... and other steps...
