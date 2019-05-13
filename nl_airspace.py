import re
import shapely
from shapely.geometry import Polygon, Point
import pandas as pd
import geopandas
import contextily as ctx
import numpy as np


def convert_fl_or_ft_asml_to_alt(line):
    if line.startswith("GND"):
        alt = 0
    elif line.startswith("FL"):
        alt = int(line[3:])*100
    elif line.endswith("ft AMSL"):
        alt = int(line[:-8])
    else:
        raise NotImplementedError("Cannot find altitude from line {0}".format(line))
    return alt


def parse_lat_lon(s):
    regex = r"(\d{1,3})°(\d{1,3})'([\d\.]{1,5})\"([NS]) (\d{1,3})°(\d{1,3})'([\d\.]{1,5})\"([EW])"
    matches = re.search(regex, s)
    if not matches:
        raise RuntimeError("Cannot find lat/lon from input: {0}".format(s))
    lat_list = matches.groups()[:4]
    lon_list = matches.groups()[4:]
    lat = float(lat_list[0]) + float(lat_list[1])/60 + float(lat_list[2])/3600
    if lat_list[3] == "S":
        lat = -1*lat
    lon = float(lon_list[0]) + float(lon_list[1]) / 60 + float(lon_list[2]) / 3600
    if lon_list[3] == "W":
        lon = -1*lon
    # Lon, lat because of x, y
    return lon, lat

# We need to account for latitude correction
def create_ctr(centre, radius_in_nm, name, airspace_class, lower, upper):
    radius = radius_in_nm/60.040
    lat = centre[1]
    circ = Point(centre).buffer(radius)
    ctr = shapely.affinity.scale(circ, 1,  np.cos(np.deg2rad(lat)))
    return {
        "name": name,
        "airspace_class": airspace_class,
        "lower_limit_ft": lower,
        "upper_limit_ft": upper,
        "type": "CTR",
        "geometry": ctr
    }

def parse_tma(s, use_3d=False):
    lines = s.splitlines()
    name = lines[0]
    airspace_class = lines[-1][-1]
    # Get info from upper/lower (lines -4 and -3)
    upper = convert_fl_or_ft_asml_to_alt(lines[-4])
    lower = convert_fl_or_ft_asml_to_alt(lines[-3])
    coordinates = []
    coordinate_strings = lines[2:-5]
    for coordinate_string in coordinate_strings:
        if coordinate_string.startswith("to point of origin"):
            # last coordinate
            coordinates.append(coordinates[0])
            break
        elif coordinate_string.startswith("along parallel to"):
            # Assumption: this doesn't really add information
            # Weird thingy of mercator projection?
            continue
        else:
            coordinates.append(parse_lat_lon(coordinate_string))
    if use_3d:
        # Appears to be useless...
        coordinates_3d = [(x, y, lower) for x, y in coordinates] + [(x, y, upper) for x, y in coordinates]
        geometry = Polygon(coordinates_3d)
    else:
        geometry = Polygon(coordinates)
    return {
        "name": name,
        "airspace_class": airspace_class,
        "lower_limit_ft": lower,
        "upper_limit_ft": upper,
        "type": "TMA",
        "geometry": geometry
    }


def add_basemap(ax, zoom, url='http://a.tile.openstreetmap.com/tileZ/tileX/tileY.png'):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))


schiphol_tma_strings = ["""Schiphol terminal control area 1

52°41'36.06"N 004°16'15.33"E;
52°48'20.00"N 005°20'00.00"E;
52°25'45.00"N 005°40'52.00"E;
52°22'41.00"N 005°40'05.00"E;
52°18'11.45"N 005°36'33.78"E;
52°11'45.00"N 005°04'24.00"E;
51°53'11.00"N 004°49'40.72"E;
51°56'10.00"N 004°21'15.00"E;
51°59'20.00"N 004°06'40.00"E;
52°17'06.01"N 003°59'10.51"E;
to point of origin.

When active, ATZ Lelystad B, glider areas Castricum 2, Hoek van Holland, and Valkenburg are excluded.

FL 095
1500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 2

52°54'00.65"N 004°07'04.77"E; 
52°48'19.15"N 004°21'00.00"E; 
52°17'06.01"N 003°59'10.51"E; 
52°17'29.93"N 003°41'47.07"E; 
to point of origin.

FL 055
3500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 3

52°11'45.00"N 005°04'24.00"E; 
52°12'18.66"N 005°07'09.97"E; 
51°53'11.00"N 005°05'47.00"E; 
along parallel to 
51°53'11.00"N 004°49'40.72"E; 
to point of origin.

FL 095
2500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 4

52°12'18.66"N 005°07'09.97"E; 
52°15'55.90"N 005°25'10.86"E; 
52°02'51.89"N 005°06'28.85"E; 
to point of origin.

FL 095
3500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 5

52°15'55.90"N 005°25'10.86"E;
52°18'11.45"N 005°36'33.78"E;
52°01'52.66"N 005°23'53.44"E;
51°53'11.00"N 005°05'47.00"E;
52°02'51.89"N 005°06'28.85"E;
to point of origin.

FL 095
FL 055

Class of airspace: A""", """Schiphol terminal control area 6

52°48'19.15"N 004°21'00.00"E; 
52°45'25.00"N 004°28'03.00"E; 
52°43'30.00"N 004°33'40.00"E; 
52°41'36.06"N 004°16'15.33"E; 
to point of origin.

FL 095
3500 ft AMSL

Class of airspace: A"""]

rotterdam_tma_strings = [
    """Rotterdam terminal control area 1

52°17'06.01"N 003°59'10.51"E;
51°59'20.00"N 004°06'40.00"E;
51°56'10.00"N 004°21'15.00"E;
51°53'11.00"N 004°49'40.72"E;
51°36'00.00"N 004°36'15.29"E;
along parallel to 
51°36'00.00"N 004°11'32.79"E;
51°37'01.21"N 004°10'42.31"E;
51°36'47.89"N 004°07'33.20"E;
51°38'00.00"N 004°05'54.80"E;
along parallel to 
51°38'00.00"N 004°04'19.30"E;
51°36'54.29"N 003°59'09.57"E;
51°37'45.27"N 003°54'51.73"E;
51°56'25.84"N 003°45'01.84"E;
to point of origin.

FL 055
1500 ft AMSL

Class of airspace: E""",
    """Rotterdam terminal control area 2

51°56'25.84"N 003°45'01.84"E;
51°37'45.27"N 003°54'51.73"E;
51°35'44.20"N 003°52'07.46"E;
51°36'16.52"N 003°50'18.27"E;
51°36'18.84"N 003°47'57.34"E;
51°35'49.85"N 003°44'48.16"E;
51°35'58.94"N 003°40'52.93"E;
51°35'06.86"N 003°37'37.99"E;
51°35'38.76"N 003°35'00.00"E;
51°35'50.00"N 003°31'10.14"E;
to point of origin.

FL 055
2500 ft AMSL

Class of airspace: E""",
    """Rotterdam terminal control area 3

52°17'29.93"N 003°41'47.07"E;
52°17'06.01"N 003°59'10.51"E;
51°35'50.00"N 003°31'10.14"E;
along parallel to 
51°35'50.00"N 003°13'49.65"E;
to point of origin.

FL 055
3500 ft AMSL

Class of airspace: E"""
]


eham_airspace = pd.DataFrame.from_records([parse_tma(tma_s) for tma_s in schiphol_tma_strings] +
                                              [create_ctr(parse_lat_lon("52°18'29.00\"N 004°45'51.00\"E"), 8, "Schiphol CTR (basic circle only)", "C", 0, 3000)])
ehrd_airspace = pd.DataFrame.from_records([parse_tma(tma_s, use_3d=True) for tma_s in rotterdam_tma_strings] +
                                     [create_ctr(parse_lat_lon("51°57'25.00\"N 004°26'14.00\"E"), 8, "Rotterdam CTR", "C", 0, 3000)])
gdf = geopandas.GeoDataFrame(ehrd_airspace).append(geopandas.GeoDataFrame(eham_airspace))

import matplotlib.pyplot as plt
plt.figure()
gdf.crs = "+init=epsg:4326"
gdf_projected = gdf.to_crs("+proj=lcc +lat_1=56.1111 +lat_2=49.5528")
ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
# add_basemap(ax, zoom=1)
# add_basemap(ax, zoom=19)
ax.set_axis_off()
plt.title("Normal")
plt.show()
plt.figure()
ax = gdf_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
# add_basemap(ax, zoom=1)
# add_basemap(ax, zoom=19)
ax.set_axis_off()
plt.title("Projected")
plt.show()


# # From https://www.wiltink.net/users/johan/notamplanner.html
# gpsSchiphol = {"lat": 52.308056, "lng": 4.764167}
# # gpsFlevo = {"lat": 52.460278, "lng": 5.527222}
# # gpsTeuge = {"lat": 52.244722, "lng": 6.046667}
# # gpsHilversum = {"lat": 52.191944, "lng": 5.146944}
# # gpsZestienhoven = {"lat": 51.956944, "lng": 4.437222}
# # gpsSeppe = {"lat": 51.554722, "lng": 4.5525}
# # gpsTexel = {"lat": 53.115278, "lng": 4.833611}
# # gpsAmeland = {"lat": 53.451667, "lng": 5.677222}
# # gpsHoogeveen = {"lat": 52.730833, "lng": 6.516111}
# # gpsEelde = {"lat": 53.125, "lng": 6.583333}
# import geopandas
#
# gpsTMA1 = [{"lat": 52.80555556, "lng": 5.33333333},
#            {"lat": 52.42916667, "lng": 5.68111111},
#            {"lat": 52.37805556, "lng": 5.66805556},
#            {"lat": 52.30318056, "lng": 5.60938333},
#            {"lat": 52.26552778, "lng": 5.41968333},
#            {"lat": 52.20518333, "lng": 5.11943611},
#            {"lat": 52.19583333, "lng": 5.07333333},
#            {"lat": 51.88638889, "lng": 4.82797778},
#            {"lat": 51.93611111, "lng": 4.35416667},
#            {"lat": 51.98888889, "lng": 4.11111111},
#            {"lat": 52.28500278, "lng": 3.98625278},
#            {"lat": 52.69335, "lng": 4.270925}]
# gpsTMA3 = [{"lat": 52.20518333, "lng": 5.11943611},
#            {"lat": 52.19583333, "lng": 5.07333333},
#            {"lat": 51.88638889, "lng": 4.82797778},
#            {"lat": 51.88638889, "lng": 5.09638889},
#            {"lat": 52.04774722, "lng": 5.10801389}]
# gpsTMA4 = [{"lat": 52.20518333, "lng": 5.11943611},
#            {"lat": 52.04774722, "lng": 5.10801389},
#            {"lat": 52.26552778, "lng": 5.41968333}]
# gpsTMA5 = [{"lat": 52.30318056, "lng": 5.60938333},
#            {"lat": 52.26552778, "lng": 5.41968333},
#            {"lat": 52.04774722, "lng": 5.10801389},
#            {"lat": 51.88638889, "lng": 5.09638889},
#            {"lat": 52.03129444, "lng": 5.39817778}]
# gpsCTR1 = [{"lat": 52.41510833, "lng": 4.63484722},
#            {"lat": 52.47255, "lng": 4.64010833},
#            {"lat": 52.46696667, "lng": 4.80336944},
#            {"lat": 52.4393, "lng": 4.80083333},
#            {"lat": 52.441005, "lng": 4.754011},
#            {"lat": 52.436966, "lng": 4.709969},
#            {"lat": 52.42892, "lng": 4.672954}]
# gpsCTR2 = [{"lat": 52.56758333, "lng": 4.60855556},
#            {"lat": 52.56764167, "lng": 4.812625},
#            {"lat": 52.46696667, "lng": 4.80336944},
#            {"lat": 52.47255, "lng": 4.64010833},
#            {"lat": 52.41510833, "lng": 4.63484722},
#            {"lat": 52.396593, "lng": 4.601527},
#            {"lat": 52.375197, "lng": 4.576129},
#            {"lat": 52.346708, "lng": 4.555843},
#            {"lat": 52.31529167, "lng": 4.54724444},
#            {"lat": 52.43069167, "lng": 4.55780278},
#            {"lat": 52.48495278, "lng": 4.57751111}]
# gpsCTR3 = [{"lat": 52.26703889, "lng": 4.5576},
#            {"lat": 52.24607, "lng": 4.571727},
#            {"lat": 52.224063, "lng": 4.595529},
#            {"lat": 52.204931, "lng": 4.626781},
#            {"lat": 52.189971, "lng": 4.663899},
#            {"lat": 52.180154, "lng": 4.704057},
#            {"lat": 52.175831, "lng": 4.739413},
#            {"lat": 52.1753, "lng": 4.779623},
#            {"lat": 52.17957222, "lng": 4.82109167},
#            {"lat": 52.06745556, "lng": 4.81079722},
#            {"lat": 52.06745556, "lng": 4.5576}]
# gpsCTRR = [{"lat": 51.84776944, "lng": 4.31397778},
#            {"lat": 51.80606111, "lng": 4.21086944},
#            {"lat": 51.88961667, "lng": 4.12254444},
#            {"lat": 51.93139444, "lng": 4.22576111},
#            {"lat": 51.903922, "lng": 4.239296},
#            {"lat": 51.879928, "lng": 4.26127},
#            {"lat": 51.864139, "lng": 4.282631}]
