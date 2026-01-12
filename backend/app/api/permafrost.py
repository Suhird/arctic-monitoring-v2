"""
Permafrost analysis API endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from geoalchemy2 import functions as geo_func
from ..database import get_db
from ..schemas.prediction import PermafrostAnalysisRequest, PermafrostAnalysisResponse
from ..services import permafrost_service
from ..utils.auth import get_current_user
from ..models.user import User
from ..models.permafrost import PermafrostData

router = APIRouter(prefix="/api/v1/permafrost", tags=["permafrost"])


@router.post("/analyze", response_model=PermafrostAnalysisResponse)
def analyze_permafrost_site(
    request: PermafrostAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze permafrost stability for a site"""
    result = permafrost_service.analyze_permafrost_stability(
        db,
        request.location,
        request.site_type,
        request.site_name,
        request.building_specs,
        str(current_user.id)
    )
    return result


@router.get("/sites")
def get_monitored_sites(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user's monitored permafrost sites"""
    sites = permafrost_service.get_user_sites(db, str(current_user.id))
    return {"sites": sites}


@router.get("/alerts")
def get_stability_alerts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get active stability alerts"""
    alerts = permafrost_service.get_alerts(db, str(current_user.id))
    return {"alerts": alerts}


@router.get("/all")
def get_all_permafrost_data(db: Session = Depends(get_db)):
    """Get all permafrost measurements from database"""
    import json
    measurements = db.query(PermafrostData).all()

    result = []
    for measurement in measurements:
        # Convert geometry to GeoJSON
        geojson = db.scalar(geo_func.ST_AsGeoJSON(measurement.location))
        location = json.loads(geojson)

        result.append({
            "id": measurement.id,
            "region_name": measurement.region_name,
            "temperature_celsius": measurement.temperature_celsius,
            "depth_meters": measurement.depth_meters,
            "measurement_date": measurement.measurement_date.isoformat() if measurement.measurement_date else None,
            "data_source": measurement.data_source,
            "location": location
        })

    return result
