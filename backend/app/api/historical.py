"""
Historical ice pattern analysis API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from ..database import get_db
from ..services import ice_analysis

router = APIRouter(prefix="/api/v1/historical", tags=["historical"])


@router.get("/patterns")
def get_ice_patterns(
    min_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    start_year: int = Query(...),
    end_year: int = Query(...),
    db: Session = Depends(get_db)
):
    """Analyze historical ice patterns"""
    bbox = (min_lon, min_lat, max_lon, max_lat)
    patterns = ice_analysis.analyze_historical_patterns(db, bbox, start_year, end_year)
    return patterns
