"""
Geospatial utility functions for PostGIS operations
"""
from typing import List, Dict, Any
from geoalchemy2.shape import to_shape
from shapely.geometry import shape, mapping
import geojson


def geometry_to_geojson(geometry) -> Dict[str, Any]:
    """Convert PostGIS geometry to GeoJSON"""
    if geometry is None:
        return None
    shapely_geom = to_shape(geometry)
    return mapping(shapely_geom)


def bbox_to_polygon_wkt(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
    """Convert bounding box coordinates to WKT POLYGON"""
    return f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"


def point_to_wkt(lon: float, lat: float) -> str:
    """Convert point coordinates to WKT POINT"""
    return f"POINT({lon} {lat})"


def linestring_to_wkt(coordinates: List[List[float]]) -> str:
    """Convert list of coordinates to WKT LINESTRING"""
    points = ", ".join([f"{coord[0]} {coord[1]}" for coord in coordinates])
    return f"LINESTRING({points})"


def create_feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a GeoJSON FeatureCollection"""
    return {
        "type": "FeatureCollection",
        "features": features
    }


def create_feature(geometry: Dict[str, Any], properties: Dict[str, Any]) -> Dict[str, Any]:
    """Create a GeoJSON Feature"""
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties
    }


def calculate_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great circle distance between two points in kilometers"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth radius in kilometers

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
