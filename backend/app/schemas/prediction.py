"""
Prediction schemas for API requests and responses
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class PredictionDayResponse(BaseModel):
    date: datetime
    type: str = "FeatureCollection"
    features: List[Dict[str, Any]]


class Prediction7DayResponse(BaseModel):
    generated_at: datetime
    predictions: List[PredictionDayResponse]


class CustomPredictionRequest(BaseModel):
    location: Dict[str, float]  # {"lat": 75.0, "lon": -120.0}
    duration_days: int
    resolution_km: Optional[float] = 10.0


class PredictionJobResponse(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime


class VesselRouteRequest(BaseModel):
    start_coords: Dict[str, float]  # {"lat": 70.0, "lon": -140.0}
    end_coords: Dict[str, float]
    vessel_type: str = "cargo"
    departure_time: Optional[datetime] = None


class VesselRouteResponse(BaseModel):
    route_id: int
    path: Dict[str, Any]  # GeoJSON LineString
    ice_risk_score: float
    estimated_duration_hours: float
    total_distance_km: float
    waypoints: List[Dict[str, Any]]


class PermafrostAnalysisRequest(BaseModel):
    location: Dict[str, float]
    site_type: str  # building, mine, infrastructure
    site_name: str
    building_specs: Optional[Dict[str, Any]] = None


class PermafrostAnalysisResponse(BaseModel):
    site_id: int
    stability_score: float
    temperature_c: float
    alert_level: str
    recommendations: List[str]
