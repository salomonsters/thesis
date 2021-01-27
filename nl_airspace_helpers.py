import re
import shapely
from shapely.geometry import Polygon, Point
import contextily as ctx
import numpy as np
from pyproj import CRS


def convert_fl_or_ft_amsl_to_alt(line):
    """
    Convert an altitude string to ft amsl. Assumes GND is 0 ft AMSL. Raises ValueError on incorrect line
    :param line: altitude string
    :return: int altitude in ft
    """
    if line.startswith("GND"):
        alt = 0
    elif line.startswith("FL"):
        alt = int(line[3:]) * 100
    elif line.endswith("ft AMSL"):
        alt = int(line[:-8])
    else:
        raise ValueError("Cannot find altitude from line {0}".format(line))
    return alt


def parse_lat_lon(s):
    """
    Parse lat-lon string from AIP to x, y pair. Note that x, y requires the return of lon, lat (so inverted!)
    Raises ValueError if unable to extract coordinates from s.

    Example: parse_lat_lon("51°57'25.00\"N 004°26'14.00\"E")
    :param s: string lat/lon from AIP.
    :return: x, y location
    """
    regex = r"(\d{1,3})°(\d{1,3})'([\d\.]{1,5})\"([NS]) (\d{1,3})°(\d{1,3})'([\d\.]{1,5})\"([EW])"
    matches = re.search(regex, s)
    if not matches:
        raise ValueError("Cannot find lat/lon from input: {0}".format(s))
    lat_list = matches.groups()[:4]
    lon_list = matches.groups()[4:]
    lat = float(lat_list[0]) + float(lat_list[1]) / 60 + float(lat_list[2]) / 3600
    if lat_list[3] == "S":
        lat = -1 * lat
    lon = float(lon_list[0]) + float(lon_list[1]) / 60 + float(lon_list[2]) / 3600
    if lon_list[3] == "W":
        lon = -1 * lon
    # Lon, lat because of x, y
    return lon, lat


def create_ctr(name, airspace_class="C", lower=0, upper=3000, centre=None, radius_in_nm=None, extra_points=None,
               geometry=None,
               set_operation=None):
    """
    Create CTR airspace dictionary

    Examples:
     normal CTR creation
     create_ctr("Rotterdam CTR", centre=parse_lat_lon("51°57'25.00\"N 004°26'14.00\"E"), radius_in_nm=8)
     CTR with extra geometry:
     create_ctr("Rotterdam CTR", centre=parse_lat_lon("51°57'25.00\"N 004°26'14.00\"E"), radius_in_nm=8, extra_points=ehrd_ctr_string, set_operation="union")
     CTR with extra geometry, difference from previous CTR:
     create_ctr("Schiphol CTR2", "C", 1200, 3000, extra_points=eham_ctr2_string, geometry=eham_ctr_1['geometry'], set_operation="difference")

    :param name: str airspace name
    :param airspace_class: str airspace class
    :param lower: number lower limit (ft)
    :param upper: number lower limit (ft)
    :param centre: x,y location of ctr centre (use parse_lat_lon on string from AIP)
    :param radius_in_nm: number radius in nm
    :param extra_points: if not just a circle, add extra geometry based on these points.
    :param geometry: if no centre/radius are given, use geometry from this parameter in set operation
    :param set_operation: set operation for the extra geometry. Options: union (geometry.union(extra_geometry from points),
    or difference which uses inverse operation (other_geometry.difference(geometry))
    :return: dict with airspace data from s, including name, airspace class, lower and upper limit in ft, type, and geometry
    """
    extra_geometry = None
    if set_operation is None and extra_points:
        set_operation = "union"
    if centre and radius_in_nm is not None:
        radius = radius_in_nm / 60.040
        lat = centre[1]
        circ = Point(centre).buffer(radius)
        # We need to account for latitude correction
        ctr = shapely.affinity.scale(circ, 1 / np.cos(np.deg2rad(lat)), 1)
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
        "ctr_centre": centre,
        "geometry": geometry
    }


def coordinatestrings2polygon(coordinate_strings):
    """
    Parse list of coordinate strings to polygon object
    :param coordinate_strings: list of coordinate strings. "to point of origin" adds first point;
    "along parallel to" is ignored.
    :return: Polygon geometry
    """
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
    """
    Parse TMA from NL AIP ocpy-pasted string
    :param s: TMA-string
    :return: dict with airspace data from s, including name, airspace class, lower and upper limit in ft, type, and geometry
    """
    lines = s.splitlines()
    name = lines[0]
    airspace_class = lines[-1][-1]
    # Get info from upper/lower (lines -4 and -3)
    upper = convert_fl_or_ft_amsl_to_alt(lines[-4])
    lower = convert_fl_or_ft_amsl_to_alt(lines[-3])
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


def add_basemap(ax, zoom, url='https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png', ll=False):
    """
    Adds basemap background to geometry. Make sure to convert geometry to epsg 3857 beforehand.

    :param ax: matplotlib axes
    :param zoom: int zoom level, usually between 5-19
    :param url: string Tile URL, with placeholders TileZ, tileX, tileY
    :param ll: bool: are coordinates assumed to be lon/lat as opposed to Spherical Mercator. Default False
    :return: None
    """
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url, ll=ll)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))
    ax.set_axis_off()


def prepare_gdf_for_plotting(gdf):
    """
    Prepare gdf for plotting by setting epsg to 3857, allowing add_basemap later.
    If no crs is set on the gdf, it assumes WGS84
    :param gdf: geopandas dataframe
    :return: None
    """
    if gdf.crs is None:
        gdf.crs = CRS("WGS84")
    return gdf.to_crs(epsg=3857)
