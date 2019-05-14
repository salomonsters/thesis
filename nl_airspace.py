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


def create_ctr(name, airspace_class, lower, upper, centre=None, radius_in_nm=None, extra_points=None, geometry=None,
               set_operation=None):
    if set_operation is None and extra_points:
        set_operation = "union"
    if centre and radius_in_nm is not None:
        radius = radius_in_nm/60.040
        lat = centre[1]
        circ = Point(centre).buffer(radius)
        # We need to account for latitude correction
        ctr = shapely.affinity.scale(circ, 1/np.cos(np.deg2rad(lat)), 1)
        geometry = ctr
    if extra_points:
        coordinate_strings = extra_points.splitlines()
        extra_geometry = coordinatestrings2polygon(coordinate_strings)
    if set_operation:
        if set_operation == "union" and extra_points:
            geometry = geometry.union(extra_geometry)
        elif set_operation == "difference" and extra_points:
            # Need other->self
            geometry = extra_geometry.difference(geometry)
        else:
            raise ValueError("Set operation {0} not valid or given without extra geometry".format(set_operation))
    return {
        "name": name,
        "airspace_class": airspace_class,
        "lower_limit_ft": lower,
        "upper_limit_ft": upper,
        "type": "CTR",
        "geometry": geometry
    }


def coordinatestrings2polygon(coordinate_strings):
    coordinates = []
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
    geometry = Polygon(coordinates)
    return geometry


def parse_tma(s):
    lines = s.splitlines()
    name = lines[0]
    airspace_class = lines[-1][-1]
    # Get info from upper/lower (lines -4 and -3)
    upper = convert_fl_or_ft_asml_to_alt(lines[-4])
    lower = convert_fl_or_ft_asml_to_alt(lines[-3])
    coordinate_strings = lines[2:-5]
    geometry = coordinatestrings2polygon(coordinate_strings)
    return {
        "name": name,
        "airspace_class": airspace_class,
        "lower_limit_ft": lower,
        "upper_limit_ft": upper,
        "type": "TMA",
        "geometry": geometry
    }


def add_basemap(ax, zoom, url='https://maps.wikimedia.org/osm-intl/tileZ/tileX/tileY.png', ll=False):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url, ll=ll)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))
    ax.set_axis_off()


from nl_airspace_def import ehrd_tma_strings, ehrd_ctr_string
from nl_airspace_def import eham_tma_strings, eham_ctr1_string, eham_ctr2_string, eham_ctr3_string

eham_ctr_1 = create_ctr("Schiphol CTR1", "C", 0, 3000,
                        centre=parse_lat_lon("52°18'29.00\"N 004°45'51.00\"E"), radius_in_nm=8,
                        extra_points=eham_ctr1_string, set_operation="union")
eham_ctr = [eham_ctr_1,
            create_ctr("Schiphol CTR2", "C", 1200, 3000, extra_points=eham_ctr2_string, geometry=eham_ctr_1['geometry'], set_operation="difference"),
            create_ctr("Schiphol CTR3", "C", 1200, 3000, extra_points=eham_ctr3_string, geometry=eham_ctr_1['geometry'], set_operation="difference")
            ]
eham_tma = [parse_tma(tma_s) for tma_s in eham_tma_strings]
eham_airspace = pd.DataFrame.from_records(eham_tma + eham_ctr)

ehrd_tma = [parse_tma(tma_s) for tma_s in ehrd_tma_strings]
ehrd_ctr = [create_ctr("Rotterdam CTR", "C", 0, 3000,
                       centre=parse_lat_lon("51°57'25.00\"N 004°26'14.00\"E"), radius_in_nm=8,
                       extra_points=ehrd_ctr_string, set_operation="union")]
ehrd_airspace = pd.DataFrame.from_records(ehrd_tma + ehrd_ctr)


ehaa_airspace = geopandas.GeoDataFrame(ehrd_airspace).append(geopandas.GeoDataFrame(eham_airspace))
ehaa_airspace.crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure()
    gdf_projected = ehaa_airspace.to_crs(epsg=3857)
    plt.figure()
    ax = gdf_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    add_basemap(ax, zoom=9, ll=False)
    plt.title("Projected")
    plt.show()

