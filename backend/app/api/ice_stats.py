"""
Ice Statistics API
Endpoints for Extent, Thickness, and Motion data
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import List, Dict, Any
from ..database import get_db
from ..models.sea_ice_extent import SeaIceExtent
from ..models.ice_thickness import IceThickness
from ..models.ice_motion import IceMotion
from geoalchemy2.shape import to_shape
import json

router = APIRouter(prefix="/api/v1/stats", tags=["ice-stats"])

@router.get("/extent")
def get_extent_history(
    days: int = Query(30, ge=7, le=3650),
    db: Session = Depends(get_db)
):
    """
    Get historical sea ice extent
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    results = db.query(SeaIceExtent).filter(
        SeaIceExtent.date >= start_date
    ).order_by(SeaIceExtent.date).all()
    
    return [
        {
            "date": r.date.strftime("%Y-%m-%d"),
            "extent_km2": r.extent_sq_km,
            "area_km2": r.area_sq_km
        }
        for r in results
    ]

@router.get("/thickness")
def get_thickness_data(
    db: Session = Depends(get_db)
):
    """
    Get latest ice thickness points as GeoJSON
    """
    # Get data from last 24 hours
    since = datetime.utcnow() - timedelta(hours=24)
    results = db.query(IceThickness).filter(
        IceThickness.timestamp >= since
    ).all()
    
    features = []
    for r in results:
        point = to_shape(r.location)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [point.x, point.y]
            },
            "properties": {
                "thickness_m": r.thickness_meters,
                "uncertainty": r.uncertainty_meters,
                "source": r.source_satellite,
                "timestamp": r.timestamp.isoformat()
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }

@router.get("/motion")
def get_motion_data(
    db: Session = Depends(get_db)
):
    """
    Get latest ice motion vectors as GeoJSON
    """
    # Get data from last 24 hours
    since = datetime.utcnow() - timedelta(hours=24)
    results = db.query(IceMotion).filter(
        IceMotion.timestamp >= since
    ).all()
    
    features = []
    for r in results:
        point = to_shape(r.location)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [point.x, point.y]
            },
            "properties": {
                "u_velocity": r.u_vector_cm_sec,
                "v_velocity": r.v_vector_cm_sec,
                "velocity_mag": (r.u_vector_cm_sec**2 + r.v_vector_cm_sec**2)**0.5,
                "source": r.source,
                "timestamp": r.timestamp.isoformat()
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }
