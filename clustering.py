import numpy as np
import scipy as sp
import pandas as pd
import itertools
import scipy as sp
import geopandas
from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import glob
import os

airspace_query = "airport=='EHRD'"
# zoom = 15
#
airspace = ehaa_airspace.query(airspace_query)

# TODO: then maybe pad as well so we don't get nan's?
# TODO: perhaps normalize to a track with a specific airspeed so we have more comparable tracks
# and then don't group on length?

def adjacency_matrix(x_i, x_j):
    x = x_i - x_j
    x.dropna(inplace=True)
    if len(x) == 0:
        return 0
    dist = np.linalg.norm(x)
    return np.exp(-dist ** 2 / (2. * (sigma * sigma)))


def scale_and_average_df(df, fields=('lon', 'lat')):
    ts_min, ts_max = df['ts'].min(), df['ts'].max()
    ts_len = ts_max - ts_min
    ts_scaled = (df['ts'] - ts_min)/ts_len*n_data_points
    df.loc[:,'timestamp'] = ts_scaled.apply(lambda x: pd.Timestamp(x, unit='s'))
    df.set_index('timestamp', inplace=True)
    resampled = df.resample(rule='10s')[fields].mean()
    return resampled


def fiedler_vector(L):
    l, U = sp.linalg.eigh(L)
    f = U[:, 1]
    return f

def spectralCluster(W, omega_min, result_indices, original_indices=None):
    if original_indices is None:
        original_indices = np.array(range(W.shape[0]))
    D = np.zeros_like(W)
    for i in range(W.shape[0]):
        D[i,i] = np.sum(W[i,:])
    L = D - W
    v = fiedler_vector(L)
    # Indices of V with positive elements
    i_l = np.ravel(np.sign(v))
    i_r = np.ravel(np.sign(-v))
    W_il_il = W[np.ix_(i_l, i_l)]
    W_ir_ir = W[np.ix_(i_r, i_r)]

    if np.var(W_il_il) > omega_min and not np.array_equal(original_indices[i_l], original_indices):
        spectralCluster(W_il_il, omega_min, result_indices, original_indices[i_l])
    else:
        result_indices.append(original_indices[i_l])
    if np.var(W_ir_ir) > omega_min and not np.array_equal(original_indices[i_r], original_indices):
        spectralCluster(W_ir_ir, omega_min, result_indices, original_indices[i_r])
    else:
        result_indices.append(original_indices[i_r])


n_data_points = 1000
sigma = 1
minalt = 1000  # ft
maxalt = 5000

# #### SINGLE FILENAME
# df = pd.read_csv('data/adsb_decoded_in_ehrd/ADSB_DECODED_20180101.csv.gz')#, converters={'callsign': lambda s: s.replace('_', '')})

# #### ENTIRE DIRECTORY
path = './data/adsb_decoded_in_ehrd/'
max_number_of_files = 5
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))[:max_number_of_files]))
import copy
orig_df = copy.deepcopy(df)

if minalt is not None:
    df = df[df['alt'] > minalt]
if maxalt is not None:
    df = df[df['alt'] < maxalt]
gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
gdf_track_converted = prepare_gdf_for_plotting(gdf)

gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y


df_g = gdf.groupby('fid')
num_groups = len(df_g)

W = np.zeros((num_groups, num_groups), dtype=float)
i = 0

fid_list = np.empty(num_groups, dtype='U10')

processed_dfs = dict()
for fid_i, df_i in df_g:
    j = 0

    if fid_i in processed_dfs:
        x_i = processed_dfs[fid_i]
    else:
        x_i = scale_and_average_df(df_i, fields=['x', 'y'])
        processed_dfs[fid_i] = x_i
        fid_list[i] = fid_i


    for fid_j, df_j in df.groupby('fid'):
        if fid_j in processed_dfs:
            x_j = processed_dfs[fid_j]
        else:
            x_j = scale_and_average_df(df_j, fields=['x', 'y'])
            processed_dfs[fid_j] = x_j
            fid_list[j] = fid_j
        Wij = adjacency_matrix(x_i, x_j)
        W[i, j] = Wij
        j += 1
    i += 1


result = []
spectralCluster(W, 0.1, result)
fid_to_cluster_map = {}
for cluster_number, fidlist_indices in enumerate(result):
    for fid in fid_list[fidlist_indices.ravel()]:
        fid_to_cluster_map[fid] = cluster_number

gdf['cluster'] = gdf['fid'].map(fid_to_cluster_map)

## VISUALISATION

df = orig_df
df['cluster'] = df['fid'].map(fid_to_cluster_map)
if minalt is not None:
    df = df[df['alt'] > minalt]
if maxalt is not None:
    df = df[df['alt'] < maxalt]

df_alt_min, df_alt_max = df['alt'].min(), df['alt'].max()
fig = plt.figure()
airspace_projected = prepare_gdf_for_plotting(airspace)
ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
# add_basemap(ax, zoom=zoom, ll=False)
ax.set_axis_off()

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
gdf_track_converted = prepare_gdf_for_plotting(gdf)
# gdf_track_converted.plot(ax=ax, column='alt', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1, vmin=df_alt_min, vmax=df_alt_max)
gdf_track_converted.plot(ax=ax, column='cluster', cmap='plasma', legend=False, markersize=0.1, linewidth=0.1)
plt.show()
