from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from ..satellite.sentinel import get_sentinel_fetcher, SentinelDataFetcher

router = APIRouter()

class RegionRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

@router.get("/sentinel/status")
async def get_sentinel_status():
    """Check if CDSE is configured and reachable"""
    fetcher = get_sentinel_fetcher()
    is_auth = fetcher._authenticate()
    return {
        "configured": bool(fetcher.client_id and fetcher.client_secret),
        "authenticated": is_auth,
        "service": "Copernicus Data Space Ecosystem"
    }

@router.post("/sentinel/analyze")
async def analyze_region_high_res(
    request: RegionRequest,
    fetcher: SentinelDataFetcher = Depends(get_sentinel_fetcher)
):
    """
    Get high-resolution ice data from real-time Sentinel imagery.
    Returns GeoJSON polygons of analyzed ice coverage.
    """
    bbox = (request.min_lon, request.min_lat, request.max_lon, request.max_lat)
    
    try:
        result = fetcher.fetch_high_res_polygons(bbox)
        if "error" in result:
             raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
def get_satellite_sources():
    """
    Get list of available satellite data sources and their status.
    """
    from ..config import settings
    
    return [
        {
            "id": "nsidc",
            "name": "NSIDC Sea Ice Index (Free)",
            "type": "passive_microwave",
            "status": "active",
            "description": "Daily sea ice concentration (~25km res). Free source.",
            "requires_key": False
        },
        {
            "id": "sentinel1",
            "name": "Sentinel-1 SAR (ESA)",
            "type": "sar",
            "status": "configured" if settings.CDSE_CLIENT_ID else "missing_credentials",
            "description": "High-res radar imagery (5-40m). Seeing through clouds/darkness.",
            "requires_key": True
        },
        {
            "id": "radarsat",
            "name": "RadarSAT (CSA)",
            "type": "sar",
            "status": "configured" if settings.RADARSAT_API_KEY else "missing_credentials",
            "description": "Canadian commercial SAR. Very high resolution.",
            "requires_key": True
        },
        {
            "id": "modis",
            "name": "NASA MODIS (Visual)",
            "type": "optical",
            "status": "active",
            "description": "Visual imagery, good for clear days.",
            "requires_key": False
        }
    ]

from fastapi.responses import StreamingResponse
import io

@router.get("/image/{product_id}")
async def get_satellite_image(
    product_id: str,
    fetcher: SentinelDataFetcher = Depends(get_sentinel_fetcher)
):
    """
    Stream the high-resolution quicklook image for a specific product.
    """
    try:
        image_bytes = fetcher.fetch_quicklook(product_id)
        if not image_bytes:
             raise HTTPException(status_code=404, detail="Image not found or fetch failed")
        
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from app.database import SessionLocal
from app.models.tile import SatelliteTile
from app.utils.tiles import tile_to_bbox, tile_to_mercator_bbox
import io
from PIL import Image
from app.satellite.maxar import get_maxar_fetcher

@router.get("/tiles/{z}/{x}/{y}")
async def get_satellite_tile(
    z: int, x: int, y: int, 
    date: str = Query(None, description="Preferred date YYYY-MM-DD"),
    source: str = Query("sentinel", description="Source: sentinel | maxar")
):
    """
    Proxy endpoint for Sentinel-1 tiles.
    Uses Database Cache for default views (Latest 7 Days).
    """
    
    # Sentinel-1 Resolution Limit Check
    # Z=3 requires >2500px request size to satisfy 1500m/px limit, which exceeds Sentinel Hub API limits.
    # We restrict to Z>=4.
    if z < 4:
        raise HTTPException(status_code=404, detail="Zoom level too low for Sentinel-1")

    # 1. Try Cache if Default View (No Date, Today, or Yesterday)
    # The cache contains the most recent mosaic (Last 7 Days), so it is valid for current views.
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    should_use_cache = False
    # Only cache Sentinel-1 tiles for now
    if source == "sentinel":
        if not date:
            should_use_cache = True
        elif date == today or date == yesterday:
            should_use_cache = True

    if should_use_cache:
        db = SessionLocal()
        try:
            cached_tile = db.query(SatelliteTile).filter_by(x=x, y=y, z=z).first()
            if cached_tile:
                return StreamingResponse(io.BytesIO(cached_tile.image), media_type="image/png")
        finally:
            db.close()
            
    # Use Web Mercator for correct projection alignment
    bbox = tile_to_mercator_bbox(x, y, z)
    
    if source == "maxar":
        # Maxar Fetching (Premium)
        fetcher = get_maxar_fetcher()
        req_size = 256
        image_bytes = fetcher.fetch_tile_image(bbox, width=req_size, height=req_size)
        
        if image_bytes:
            return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
        else:
             return StreamingResponse(io.BytesIO(b""), media_type="image/png")

    # Time Range logic
    if date:
        # User selected a specific date. 
        # Sentinel-1 revisit is ~3-6 days. A 24h window (single date) often results in empty maps.
        # We expand the window to look back 3 days from the selected date to create a mosaic.
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            start = target_date - timedelta(days=5)
            end = target_date + timedelta(days=1) 
            
            start_time = start.strftime("%Y-%m-%dT00:00:00Z")
            end_time = target_date.strftime("%Y-%m-%dT23:59:59Z")
            
            # print(f"Tile Req (Live): {z}/{x}/{y} [Window: {start_time} to {end_time}]")
        except ValueError:
             # Fallback
             start_time = f"{date}T00:00:00Z"
             end_time = f"{date}T23:59:59Z"
    else:
        # Last 7 days to match the Cache Seed logic (if cache miss)
        now = datetime.utcnow()
        start = now - timedelta(days=7)
        end_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Tile Req (Cache Miss): {z}/{x}/{y} [Default: Last 7 Days]")
        
    fetcher = get_sentinel_fetcher()
    
    # Dynamic Resolution to valid Sentinel Hub limits
    # Limit is 1500m/px.
    # Z=4 at some lats gave 1623m/px with 1024px.
    # 2048px will give ~800m/px, which is valid.
    req_size = 256
    if z < 7:
        req_size = 2048
    
    try:
        image_bytes = fetcher.fetch_tile_image(
            bbox, 
            (start_time, end_time), 
            width=req_size, 
            height=req_size,
            crs="http://www.opengis.net/def/crs/EPSG/0/3857"
        )
        
        # Resize if needed
        if image_bytes and req_size > 256:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()

    except Exception as e:
        print(f"Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Fetch failed")
    
    if not image_bytes:
        raise HTTPException(status_code=404, detail="No data")
        
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@router.get("/products")
async def list_satellite_products(
    min_lon: float = Query(-180, ge=-180, le=180),
    min_lat: float = Query(60, ge=-90, le=90),
    max_lon: float = Query(180, ge=-180, le=180),
    max_lat: float = Query(90, ge=-90, le=90),
    start_date: str = Query(None, description="YYYY-MM-DD"),
    end_date: str = Query(None, description="YYYY-MM-DD"),
    fetcher: SentinelDataFetcher = Depends(get_sentinel_fetcher)
):
    """
    List available satellite products for a given region without downloading images.
    """
    try:
        if not fetcher._authenticate():
             raise HTTPException(status_code=401, detail="Authentication failed")

        s_date = datetime.utcnow() - timedelta(days=3)
        e_date = datetime.utcnow()
        
        if start_date:
             s_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
             e_date = datetime.strptime(end_date, "%Y-%m-%d")
             
        bbox = (min_lon, min_lat, max_lon, max_lat)
        products = fetcher.search_products(bbox, s_date, e_date, collection="SENTINEL-1")
        
        return products
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
