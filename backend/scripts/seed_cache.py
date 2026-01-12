
import sys
import os
import math
from datetime import datetime, timedelta
import logging

# Add backend to path
sys.path.append(os.getcwd())

from app.utils.tiles import tile_to_bbox, tile_to_mercator_bbox
from app.database import init_db, SessionLocal
from app.models.tile import SatelliteTile
from app.satellite.sentinel import get_sentinel_fetcher

from PIL import Image
import io
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def latlon_to_tile(lat, lon, z):
    n = 2.0 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    # Handle poles
    try:
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    except ValueError:
        ytile = 0
    return xtile, ytile

def get_tiles_in_bbox(bbox, z):
    min_lon, min_lat, max_lon, max_lat = bbox
    # Top Left (using maxLat for minY due to inverted Y)
    min_x, min_y = latlon_to_tile(max_lat, min_lon, z)
    # Bottom Right
    max_x, max_y = latlon_to_tile(min_lat, max_lon, z)
    
    tiles = []
    # Ensure range order
    start_x, end_x = min(min_x, max_x), max(min_x, max_x)
    start_y, end_y = min(min_y, max_y), max(min_y, max_y)

    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            tiles.append((x, y, z))
    return tiles

def seed_cache():
    logger.info("Initializing Database...")
    init_db() # Ensure tables exist
    
    db = SessionLocal()
    fetcher = get_sentinel_fetcher()
    if not fetcher._authenticate():
        logger.error("Failed to authenticate with Sentinel Hub")
        return

    # Define Arctic Region BBox (Lat > 60)
    arctic_bbox = [-180, 60, 180, 85] 

    # Time Window: Last 7 Days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    time_range = (start_date.strftime("%Y-%m-%dT%H:%M:%SZ"), 
                  end_date.strftime("%Y-%m-%dT%H:%M:%SZ"))
    
    logger.info(f"Seeding Cache for Time Range: {time_range}")
    
    # Process Zoom Levels 4, 5 (Z=3 is too low res for S1 limits)
    for z in [4, 5]:
        tiles = get_tiles_in_bbox(arctic_bbox, z)
        logger.info(f"Processing Zoom {z}: {len(tiles)} tiles found in Arctic.")
        
        req_size = 256
        if z < 7:
            req_size = 2048
            
        count = 0
        # Use TQDM for progress tracking (Unit: tiles)
        for x, y, z in tqdm(tiles, desc=f"Processing Zoom {z}", unit="tile"):
            bbox = tile_to_mercator_bbox(x, y, z)
            
            try:
                # Fetch with high res if needed
                image_bytes = fetcher.fetch_tile_image(
                    bbox, 
                    time_range, 
                    width=req_size, 
                    height=req_size,
                    crs="http://www.opengis.net/def/crs/EPSG/0/3857"
                )
                
                if image_bytes:
                    # Resize if high res was requested
                    if req_size > 256:
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            img = img.resize((256, 256), Image.Resampling.LANCZOS)
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            image_bytes = buf.getvalue()
                    
                    # Upsert
                    existing = db.query(SatelliteTile).filter_by(x=x, y=y, z=z).first()
                    if existing:
                        existing.image = image_bytes
                        existing.last_updated = datetime.utcnow()
                    else:
                        new_tile = SatelliteTile(x=x, y=y, z=z, image=image_bytes)
                        db.add(new_tile)
                    
                    if count % 10 == 0:
                        db.commit() # Commit batch
                    
                    count += 1
                else:
                    pass
                    
            except Exception as e:
                logger.error(f"Error processing tile {z}/{x}/{y}: {e}")
                
        db.commit()
        logger.info(f"\nZoom {z} complete. Updated {count} tiles.")


    db.close()
    logger.info("Cache Seeding Complete.")

if __name__ == "__main__":
    seed_cache()
