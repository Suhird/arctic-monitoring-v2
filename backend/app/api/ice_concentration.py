"""
Ice concentration API endpoints
"""
from fastapi import APIRouter, Depends, Query, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from ..database import get_db
from ..schemas.ice import IceConcentrationResponse, IceTimeSeriesResponse
from ..services import ice_analysis
from ..services import ice_analysis
from ..ml.model_loader import model_loader
import logging
import numpy as np
from PIL import Image
import io

router = APIRouter(prefix="/api/v1/ice", tags=["ice-concentration"])
logger = logging.getLogger(__name__)


    # Initialize Clients (Sentinel is loaded lazily to avoid circular imports if possible)
    # nsidc_client removed as per deprecation plan

    @router.get("/current", response_model=IceConcentrationResponse)
    def get_current_ice_data(
        min_lon: float = Query(..., ge=-180, le=180),
        min_lat: float = Query(..., ge=-90, le=90),
        max_lon: float = Query(..., ge=-180, le=180),
        max_lat: float = Query(..., ge=-90, le=90),
        source: Optional[str] = Query(None, description="Satellite source (sentinel1, radarsat)"),
        date: Optional[str] = Query(None, description="Target date (YYYY-MM-DD)"),
        db: Session = Depends(get_db)
    ):
        """
        Get current ice concentration data for a bounding box
        Cached for 5 minutes
        """
        bbox = (min_lon, min_lat, max_lon, max_lat)
        
        # Parse date
        target_date_obj = None
        if date:
            try:
                target_date_obj = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                pass # Use latest if invalid

        result = ice_analysis.get_current_ice_concentration(
            db, 
            bbox, 
            source=source,
            target_date=target_date_obj
        )
        return result


    @router.get("/timeseries")
    def get_ice_timeseries(
        min_lon: float = Query(..., ge=-180, le=180),
        min_lat: float = Query(..., ge=-90, le=90),
        max_lon: float = Query(..., ge=-180, le=180),
        max_lat: float = Query(..., ge=-90, le=90),
        start_date: datetime = Query(...),
        end_date: datetime = Query(...),
        db: Session = Depends(get_db)
    ):
        """Get historical ice concentration time series"""
        bbox = (min_lon, min_lat, max_lon, max_lat)
        result = ice_analysis.get_ice_time_series(db, bbox, start_date, end_date)
        return {
            "bbox": list(bbox),
            "data": result
        }


    @router.get("/realtime")
    def get_realtime_satellite_data(
        min_lon: float = Query(-180, ge=-180, le=180),
        min_lat: float = Query(60, ge=-90, le=90),
        max_lon: float = Query(180, ge=-180, le=180),
        max_lat: float = Query(90, ge=-90, le=90),
        date: str = Query(None, description="Date in YYYY-MM-DD format"),
        source: Optional[str] = Query(None, description="Satellite source (sentinel1, radarsat)"),
        db: Session = Depends(get_db)
    ):
        """
        Get real-time ice concentration data.
        Priority:
        1. Database (Bremen)
        2. Sentinel-1/2 API (live fetch)
        3. Fallback (Simulated)
        """
        from ..satellite.sentinel import get_sentinel_fetcher
        
        bbox_tuple = (min_lon, min_lat, max_lon, max_lat)
        
        # Parse date if provided
        target_date = None
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                # If invalid date, ignore or raise error. 
                # For robustness, we'll log warning and ignore 
                logger.warning(f"Invalid date format received: {date}")

        try:
            # If specific source requested, strict fetch
            if source:
                logger.info(f"Fetching data strictly for source: {source}")
                
                if source == 'sentinel1':
                    fetcher = get_sentinel_fetcher()
                    fetcher._authenticate()
                    return fetcher.fetch_high_res_polygons(bbox_tuple)
                    
                elif source == 'radarsat':
                    from ..config import settings
                    if not settings.RADARSAT_API_KEY:
                        # Soft fail if no key, as requested
                        logger.warning("RadarSAT requested but no API key configured.")
                        return {
                            "type": "FeatureCollection",
                            "features": [],
                            "metadata": {"warning": "RadarSAT API key not configured"}
                        }
                    # Placeholder for actual RadarSAT implementation
                    # In a real app, we would call the CSA API here
                    logger.info("Fetching RadarSAT data (Mocking for MVP as API access is limited)")
                    from ..services.mock_data import generate_mock_ice_data
                    return generate_mock_ice_data(bbox_tuple, source="radarsat")
                    
                else:
                    # Generic fallback or error
                    pass

            # Default Behavior (Auto-Selection)
            # 1. Check Database for recent data (Bremen)
            db_data = ice_analysis.get_current_ice_concentration(
                db, 
                bbox_tuple, 
                limit=10000, 
                target_date=target_date
            )
            
            if db_data["type"] == "FeatureCollection" and len(db_data["features"]) > 0:
                logger.info(f"✅ Serving {len(db_data['features'])} features from database.")
                return db_data

            # 2. Skip NSIDC

            # 3. Fallback to Sentinel (High res)
            logger.info(f"DB empty, checking Sentinel.")
            try:
                fetcher = get_sentinel_fetcher()
                fetcher._authenticate()
                geojson = fetcher.fetch_high_res_polygons(bbox_tuple)
                if "error" not in geojson:
                    return geojson
            except Exception as e:
                logger.warning(f"Sentinel fetch failed: {e}")

        # 4. Final Fallback
        logger.info("Generating simulated ice data for fallback.")
        return generate_mock_ice_data(bbox_tuple)

    except Exception as e:
        logger.error(f"Error fetching satellite data: {e}")
        # Return empty or demo data on error to prevent crash
        return {"type": "FeatureCollection", "features": [], "error": str(e)}


@router.post("/classify")
async def classify_ice_image(
    file: UploadFile = File(..., description="Satellite image file (PNG, JPG, GeoTIFF)")
):
    """
    Classify ice type from uploaded satellite image using ResNet50 model

    Returns:
        - ice_type: Predicted ice type (open_water, thin_ice, thick_ice)
        - confidence: Classification confidence (0-100%)
        - probabilities: Probabilities for all classes
    """
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array
        image_array = np.array(image)

        logger.info(f"Classifying image: {file.filename}, shape: {image_array.shape}")

        # Load classifier model
        classifier = model_loader.get_ice_classifier()

        # Run prediction
        result = classifier.predict(image_array)

        logger.info(f"Classification result: {result}")

        return {
            "filename": file.filename,
            "ice_type": result["ice_type"],
            "confidence": result["confidence"] * 100,  # Convert to percentage
            "probabilities": {k: v * 100 for k, v in result["probabilities"].items()},
            "model_version": "resnet50-v1.0"
        }

    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")


@router.post("/changes")
async def detect_ice_changes(
    image1: UploadFile = File(..., description="First satellite image (earlier time)"),
    image2: UploadFile = File(..., description="Second satellite image (later time)")
):
    """
    Detect changes in ice coverage between two satellite images using Siamese Network

    Returns:
        - mean_change: Average change probability across the image (0-100%)
        - max_change: Maximum change probability (0-100%)
        - changed_area_percent: Percentage of pixels with >50% change probability
        - change_map: 2D array of change probabilities (optional, can be large)
    """
    try:
        # Read both images
        contents1 = await image1.read()
        contents2 = await image2.read()

        img1 = Image.open(io.BytesIO(contents1))
        img2 = Image.open(io.BytesIO(contents2))

        # Convert to RGB if needed
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')

        # Convert to numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        logger.info(f"Detecting changes between {image1.filename} and {image2.filename}")
        logger.info(f"Image 1 shape: {img1_array.shape}, Image 2 shape: {img2_array.shape}")

        # Load change detector model
        from ..ml.change_detector import SiameseChangeDetector
        import torch
        import os
        from ..config import settings

        model_path = os.path.join(settings.MODEL_PATH, settings.CHANGE_DETECTOR_MODEL)
        change_detector = SiameseChangeDetector()

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                change_detector.load_state_dict(checkpoint['model_state_dict'])
            else:
                change_detector.load_state_dict(checkpoint)
            change_detector.eval()
            logger.info(f"Change detector loaded from {model_path}")

        # Run change detection
        change_map = change_detector.detect_changes(img1_array, img2_array, threshold=0.5)

        # Calculate statistics
        mean_change = float(change_map.mean())
        max_change = float(change_map.max())
        changed_pixels = int((change_map > 0).sum())
        total_pixels = change_map.size
        changed_area_percent = (changed_pixels / total_pixels) * 100

        logger.info(f"Change detection complete: {changed_area_percent:.2f}% area changed")

        return {
            "image1": image1.filename,
            "image2": image2.filename,
            "mean_change_probability": mean_change * 100,
            "max_change_probability": max_change * 100,
            "changed_area_percent": changed_area_percent,
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "model_version": "siamese-v1.0",
            # Note: change_map array not included to reduce response size
            # Add "change_map": change_map.tolist() if needed
        }

    except Exception as e:
        logger.error(f"Error detecting changes: {e}")
        raise HTTPException(status_code=500, detail=f"Change detection failed: {str(e)}")
