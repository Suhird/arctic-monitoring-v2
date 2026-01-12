"""
Vessel routing API endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from geoalchemy2 import functions as geo_func
from ..database import get_db
from ..schemas.prediction import VesselRouteRequest, VesselRouteResponse
from ..services import routing_service
from ..utils.auth import get_current_user
from ..models.user import User
from ..models.vessel_route import VesselRoute

router = APIRouter(prefix="/api/v1", tags=["routing"])


@router.post("/routing/calculate", response_model=VesselRouteResponse)
def recommend_vessel_route(
    request: VesselRouteRequest,
    db: Session = Depends(get_db)
):
    """Get optimal vessel route considering ice conditions"""
    result = routing_service.calculate_optimal_route(
        db,
        request.start_coords,
        request.end_coords,
        request.vessel_type,
        request.departure_time,
        None  # user_id is optional
    )
    return result


@router.get("/routes/all")
def get_all_routes(db: Session = Depends(get_db)):
    """Get all vessel routes from database"""
    routes = db.query(VesselRoute).all()

    result = []
    for route in routes:
        # Convert geometry to GeoJSON
        geojson = db.scalar(geo_func.ST_AsGeoJSON(route.route_geometry))

        import json
        geometry = json.loads(geojson)

        result.append({
            "id": route.id,
            "vessel_id": route.vessel_id,
            "route_name": route.route_name,
            "distance_km": route.distance_km,
            "estimated_duration_hours": route.estimated_duration_hours,
            "ice_risk_score": route.ice_risk_score,
            "geometry": geometry
        })

    return result
