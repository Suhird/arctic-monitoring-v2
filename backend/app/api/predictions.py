"""
Ice prediction API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from geoalchemy2 import functions as geo_func
from ..database import get_db
from ..services import prediction_service
from ..models.prediction import IcePrediction
import json

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.get("/7day")
def get_7day_prediction(
    min_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    db: Session = Depends(get_db),
    use_ml: bool = Query(True, description="Use ML model for predictions (True) or database demo data (False)")
):
    """
    Get 7-day ice prediction data - now using real LSTM model predictions!
    Set use_ml=False to use database demo data instead.
    """
    if use_ml:
        # Use real ML predictions from LSTM model
        bbox = (min_lon, min_lat, max_lon, max_lat)
        ml_predictions = prediction_service.generate_7day_prediction(db, bbox)

        # Convert to GeoJSON format for frontend
        # Create grid-based features from spatial data
        features = []
        for day_pred in ml_predictions:
            # Create a simple polygon for the bounding box
            # In production, you'd create grid cells or polygons from spatial_data
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat]
                    ]]
                },
                "properties": {
                    "forecast_days": day_pred.get("day", 1),
                    "predicted_concentration": day_pred.get("predicted_concentration", 0),
                    "confidence_score": day_pred.get("confidence", 0.75),
                    "model_version": "lstm-v1.0-ml",
                    "prediction_date": day_pred.get("date")
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }
    else:
        # Use database demo data (old behavior)
        predictions = db.query(IcePrediction).all()

        features = []
        for pred in predictions:
            # Convert geometry to GeoJSON
            geojson = db.scalar(geo_func.ST_AsGeoJSON(pred.geometry))
            geometry = json.loads(geojson)

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "forecast_days": pred.forecast_days,
                    "predicted_concentration": pred.predicted_concentration,
                    "confidence_score": pred.confidence_score,
                    "model_version": pred.model_version,
                    "prediction_date": pred.prediction_date.isoformat() if pred.prediction_date else None
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }


@router.get("/ml-forecast")
def get_ml_forecast(
    min_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    db: Session = Depends(get_db)
):
    """
    Get ML-generated 7-day ice movement forecast
    Cached for 1 hour
    """
    bbox = (min_lon, min_lat, max_lon, max_lat)
    predictions = prediction_service.generate_7day_prediction(db, bbox)

    return {
        "bbox": list(bbox),
        "predictions": predictions
    }
