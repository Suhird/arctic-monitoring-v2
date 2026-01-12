
import math

def tile_to_bbox(x: int, y: int, z: int):
    """
    Convert XYZ tile coordinates to WGS84 bounding box [min_lon, min_lat, max_lon, max_lat].
    """
    def tile_to_lon(x, z):
        return x / math.pow(2, z) * 360.0 - 180.0

    def tile_to_lat(y, z):
        n = math.pi - 2.0 * math.pi * y / math.pow(2, z)
        return 180.0 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

    min_lon = tile_to_lon(x, z)
    max_lon = tile_to_lon(x + 1, z)
    min_lat = tile_to_lat(y + 1, z)
    max_lat = tile_to_lat(y, z)

    return [min_lon, min_lat, max_lon, max_lat]

def tile_to_mercator_bbox(x: int, y: int, z: int):
    """
    Convert XYZ tile coordinates to Web Mercator (EPSG:3857) bounding box [min_x, min_y, max_x, max_y].
    Internal helper for projection-accurate requests.
    """
    EARTH_CIRCUMFERENCE = 40075016.68557849
    ORIGIN_SHIFT = EARTH_CIRCUMFERENCE / 2.0
    
    tile_size = EARTH_CIRCUMFERENCE / math.pow(2, z)
    
    min_x = -ORIGIN_SHIFT + x * tile_size
    max_x = -ORIGIN_SHIFT + (x + 1) * tile_size
    
    # Y is inverted (Top=MaxY)
    max_y = ORIGIN_SHIFT - y * tile_size
    min_y = ORIGIN_SHIFT - (y + 1) * tile_size
    
    return [min_x, min_y, max_x, max_y]
