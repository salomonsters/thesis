import numpy as np
import pandas as pd
import geopandas
from nl_airspace_def import ehaa_airspace, hoofddorp_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import glob
import os

airspace_query = "airport=='EHAM'"
zoom = 14
# fid_to_cluster_map = {'194c4d72': 0, '29aa7d4c': 0, '2cfbceec': 0, '301e06d0': 0, '4af38dcc': 0, '4b1e4256': 0, '61be46c8': 0, '7ba88512': 0, '8461eef0': 0, '8765ef02': 0, '8e4ec960': 0, 'a69af8e0': 0, 'ccac88b4': 0, 'd32ccd66': 0, 'f0ee30ac': 0, '38843042': 1, '45762be8': 1, '54b40de6': 1, '790f732e': 1, '803f8148': 1, '9ab7b0f4': 1, 'a333e054': 1, 'd8672b6e': 1, 'd9d74354': 1, 'dbbe68f0': 1, 'dc98a046': 1, 'e01a9766': 1, 'e5557b96': 1, 'e8287a4a': 1, 'ef0685be': 1, 'fa2d46c6': 1}


airspace = ehaa_airspace.query(airspace_query)

fids = ['01cf3c72', '03adb898', '056b3610', '05df1fbc', '14bcd650',
       '1d59b4fe', '1dba7bae', '2c587800', '4057f434', '419ae7d4',
       '505ba72c', '56e809f0', '7ba88512', '89066a80', '9d5aadca',
       '9d9dcc36', '9df2dcd0', 'a189653a', 'b879654c', 'c2b7226a',
       'd492403c', 'd65ddc00', 'd71634a8', 'd882bb54', 'fb0895aa',
       'ff616302']
df = pd.read_csv('data/adsb_decoded_in_eham/ADSB_DECODED_20180101.csv.gz')#, converters={'callsign': lambda s: s.replace('_', '')})
df = df.query("fid in @fids")
# df['cluster'] = df['fid'].map(fid_to_cluster_map)
## ENTIRE DIRECTORY
# path = './data/adsb_decoded_in_hoofddorp/'
# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))))
df_alt_min, df_alt_max = 200, 6000 # df['alt'].min(), df['alt'].max()
fig = plt.figure()
airspace_projected = prepare_gdf_for_plotting(airspace)
ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
# add_basemap(ax, zoom=zoom, ll=False)
ax.set_axis_off()

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
gdf_track_converted = prepare_gdf_for_plotting(gdf)
joined = geopandas.sjoin(gdf_track_converted, airspace_projected, how="inner", op="intersects")
joined.plot(ax=ax, column='alt', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1, vmin=df_alt_min, vmax=df_alt_max)
# gdf_track_converted.plot(ax=ax, column='cluster', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1)
plt.show()
