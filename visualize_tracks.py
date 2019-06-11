import numpy as np
import pandas as pd
import geopandas
from nl_airspace_def import ehaa_airspace, hoofddorp_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import glob
import os

# airspace_query = "airport=='EHRD' and type=='CTR'"
zoom = 14
# fid_to_cluster_map = {'194c4d72': 0, '29aa7d4c': 0, '2cfbceec': 0, '301e06d0': 0, '4af38dcc': 0, '4b1e4256': 0, '61be46c8': 0, '7ba88512': 0, '8461eef0': 0, '8765ef02': 0, '8e4ec960': 0, 'a69af8e0': 0, 'ccac88b4': 0, 'd32ccd66': 0, 'f0ee30ac': 0, '38843042': 1, '45762be8': 1, '54b40de6': 1, '790f732e': 1, '803f8148': 1, '9ab7b0f4': 1, 'a333e054': 1, 'd8672b6e': 1, 'd9d74354': 1, 'dbbe68f0': 1, 'dc98a046': 1, 'e01a9766': 1, 'e5557b96': 1, 'e8287a4a': 1, 'ef0685be': 1, 'fa2d46c6': 1}


airspace = hoofddorp_airspace # ehaa_airspace.query(airspace_query)


## SINGLE FILENAME
# df = pd.read_csv('data/adsb_decoded_in_hoofddorp/ADSB_DECODED_20180101.csv.gz')#, converters={'callsign': lambda s: s.replace('_', '')})
# df['cluster'] = df['fid'].map(fid_to_cluster_map)
## ENTIRE DIRECTORY
path = './data/adsb_decoded_in_hoofddorp/'
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))))
df_alt_min, df_alt_max = 0, 6000 # df['alt'].min(), df['alt'].max()
fig = plt.figure()
airspace_projected = prepare_gdf_for_plotting(airspace)
ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
add_basemap(ax, zoom=zoom, ll=False)
ax.set_axis_off()

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
gdf_track_converted = prepare_gdf_for_plotting(gdf)
joined = geopandas.sjoin(gdf_track_converted, airspace_projected, how="inner", op="intersects")
joined.plot(ax=ax, column='alt', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1, vmin=df_alt_min, vmax=df_alt_max)
# gdf_track_converted.plot(ax=ax, column='cluster', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1)
plt.show()
