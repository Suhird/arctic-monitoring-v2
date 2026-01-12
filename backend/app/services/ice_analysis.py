"""
Ice data analysis service
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from geoalchemy2.functions import ST_Intersects, ST_GeomFromText
from datetime import datetime, timedelta
from typing import List, Dict, Any
from ..models.ice_data import IceConcentration
from ..utils.geospatial import geometry_to_geojson, create_feature_collection, create_feature
from ..redis_client import redis_client
import hashlib
import json


def get_current_ice_concentration(
    db: Session,
    bbox: tuple,
    limit: int = 1000,
    exclude_sources: List[str] = None,
    source: str = None,
    target_date: datetime.date = None
) -> Dict[str, Any]:
    """
    Get current ice concentration data for a bounding box
    
    Args:
        db: Database session
        bbox: (min_lon, min_lat, max_lon, max_lat)
        limit: Maximum number of results
        exclude_sources: List of satellite sources to exclude
        source: Specific source to include (e.g. 'sentinel1', 'nsidc')
    """
    # Create cache key
    bbox_hash = hashlib.md5(str(bbox).encode()).hexdigest()
    if source:
        cache_key = f"ice:current:{bbox_hash}:source:{source}"
    elif exclude_sources:
        exclude_str = ",".join(sorted(exclude_sources))
        exclude_hash = hashlib.md5(exclude_str.encode()).hexdigest()
        cache_key = f"ice:current:{bbox_hash}:{exclude_hash}"
    else:
        cache_key = f"ice:current:{bbox_hash}"
    
    if target_date:
        cache_key += f":{target_date.isoformat()}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return cached

    # Query database
    # Import the NEW model
    from ..models.ingestion import IceConcentrationDaily
    
    bbox_wkt = f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"

    query = db.query(IceConcentrationDaily).filter(
        ST_Intersects(IceConcentrationDaily.geometry, ST_GeomFromText(bbox_wkt, 4326))
    )

    if target_date:
        query = query.filter(func.date(IceConcentrationDaily.date) == target_date)
    else:
        # Default: Get most recent data available
        # Find max date first
        latest_date = db.query(func.max(IceConcentrationDaily.date)).scalar()
        if latest_date:
            query = query.filter(func.date(IceConcentrationDaily.date) == latest_date.date())

    if source:
        query = query.filter(IceConcentrationDaily.source == source)
    elif exclude_sources:
        query = query.filter(IceConcentrationDaily.source.notin_(exclude_sources))

    ice_data = query.limit(limit).all()

    # Convert to GeoJSON
    features = []
    for ice in ice_data:
        # geometry_to_geojson might expect a specific shape or object
        # safely convert wkt/geojson
        geom_json = geometry_to_geojson(ice.geometry)
        
        feature = create_feature(
            geom_json,
            {
                "concentration": ice.concentration, # 0-100
                "source": ice.source,
                "timestamp": ice.date.isoformat(),
                "resolution_km": ice.resolution_km
            }
        )
        features.append(feature)

    result = create_feature_collection(features)

    # Cache for 5 minutes
    redis_client.set(cache_key, result, ttl=300)

    return result


def get_ice_time_series(
    db: Session,
    bbox: tuple,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """Get time series of ice concentration for an area"""
    bbox_wkt = f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"

    ice_data = db.query(IceConcentration).filter(
        and_(
            ST_Intersects(IceConcentration.geometry, ST_GeomFromText(bbox_wkt, 4326)),
            IceConcentration.timestamp >= start_date,
            IceConcentration.timestamp <= end_date
        )
    ).order_by(IceConcentration.timestamp).all()

    # Aggregate by day
    daily_data = {}
    for ice in ice_data:
        date_key = ice.timestamp.date().isoformat()
        if date_key not in daily_data:
            daily_data[date_key] = []
        daily_data[date_key].append({
            "concentration_percent": ice.concentration_percent,
            "ice_type": ice.ice_type
        })

    # Calculate daily averages
    time_series = []
    for date_str, data_list in sorted(daily_data.items()):
        avg_concentration = sum(d["concentration_percent"] for d in data_list) / len(data_list)
        time_series.append({
            "date": date_str,
            "avg_concentration": avg_concentration,
            "sample_count": len(data_list)
        })

    return time_series


def analyze_historical_patterns(
    db: Session,
    bbox: tuple,
    start_year: int,
    end_year: int
) -> Dict[str, Any]:
    """Analyze historical ice patterns"""
    # This would perform complex analysis
    # Placeholder implementation
    return {
        "seasonal_patterns": {
            "winter_avg": 85.0,
            "spring_avg": 65.0,
            "summer_avg": 35.0,
            "fall_avg": 55.0
        },
        "trends": {
            "annual_change_percent": -2.3,
            "direction": "declining"
        },
        "statistics": {
            "min_concentration": 15.0,
            "max_concentration": 95.0,
            "avg_concentration": 60.0
        }
    }
