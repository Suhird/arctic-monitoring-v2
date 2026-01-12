import os
import sys
import random
import logging
import json
import urllib.request
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from geoalchemy2.shape import from_shape
from shapely.geometry import shape, Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base, settings
from app.models.ice_data import IceConcentration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LAND_GEOJSON_URL = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_land.geojson"

def get_db_session():
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def download_land_polygons():
    logger.info("Downloading land polygons...")
    with urllib.request.urlopen(LAND_GEOJSON_URL) as url:
        data = json.loads(url.read().decode())
    
    geoms = []
    for feature in data['features']:
        s = shape(feature['geometry'])
        if s.is_valid:
            geoms.append(s)
        else:
            geoms.append(s.buffer(0)) # Fix invalid geometries
    
    return unary_union(geoms)

def generate_blob_field(center_lat, center_lon, num_blobs, radius_range, lon_spread=30):
    """Generates a merged polygon from random blobs to create organic shapes."""
    blobs = []
    for _ in range(num_blobs):
        # Randomize position
        lat = center_lat + random.uniform(-10, 10)
        # Handle longitude wrapping logic simply by allowing -180 to 180 generation
        if lon_spread >= 360:
             lon = random.uniform(-180, 180)
        else:
             lon = center_lon + random.uniform(-lon_spread, lon_spread)
        
        # Create a point and buffer it to make a circle (blob)
        # Adjusting radius for latitude distortion approx
        radius = random.uniform(*radius_range)
        blob = Point(lon, lat).buffer(radius)
        blobs.append(blob)
    
    return unary_union(blobs)

def generate_mock_data(session):
    """
    Generate realistic organic ice data for the ENTIRE Arctic Ocean with land masking.
    """
    logger.info("Generating global organic ice concentration data...")
    
    timestamp = datetime.utcnow()
    
    # 1. Download Land Mask
    land_geom = download_land_polygons()
    logger.info("Land mask prepared.")

    # 2. Clear existing data
    session.query(IceConcentration).filter(
        IceConcentration.satellite_source.in_(['mock_copernicus', 'mock_organic'])
    ).delete()
    
    features = []

    # 3. Generate Ice Layers (Concentration Bands)
    # Global coverage strategy: Generate blobs around the pole across all longitudes
    
    # Layer 1: Thick Multi-year Ice (Central Arctic) - 90-100%
    # Centered on Pole, spread across all longitudes
    thick_ice_raw = generate_blob_field(86, 0, 150, (3, 15), lon_spread=360)
    
    # Layer 2: Pack Ice (Mid Arctic) - 70-90%
    # Concentric ring somewhat, but represented by more blobs spread 360
    pack_ice_raw = generate_blob_field(80, 0, 250, (3, 12), lon_spread=360)
    
    # Layer 3: Marginal Ice Zone (Lower Arctic) - 40-70%
    # Further south, extending into seas
    marginal_ice_raw = generate_blob_field(72, 0, 300, (3, 10), lon_spread=360)
    
    # 4. Refine Layers (Boolean Operations)
    
    # Clip everything to valid Northern Hemisphere latitudes above 60
    arctic_box = box(-180, 60, 180, 90)
    
    thick_ice = thick_ice_raw.intersection(arctic_box)
    pack_ice = pack_ice_raw.intersection(arctic_box).difference(thick_ice)
    marginal_ice = marginal_ice_raw.intersection(arctic_box).difference(pack_ice).difference(thick_ice)
    
    layers = [
        (thick_ice, 95.0, "multi_year"),
        (pack_ice, 80.0, "pack_ice"),
        (marginal_ice, 50.0, "first_year")
    ]
    
    count = 0
    for geom_layer, concentration, ice_type in layers:
        if geom_layer.is_empty:
            continue
            
        # 5. Mask Land
        # Subtract land from the ice layer
        water_ice = geom_layer.difference(land_geom)
        
        if water_ice.is_empty:
            continue

        # Handle MultiPolygons by splitting them
        polys = []
        if isinstance(water_ice, MultiPolygon):
            polys.extend(water_ice.geoms)
        else:
            polys.append(water_ice)
            
        for poly in polys:
            if poly.area < 0.1: # Skip tiny artifacts
                continue
                
            # Simplify slightly to reduce complexity
            simple_poly = poly.simplify(0.05)
            
            ice_record = IceConcentration(
                timestamp=timestamp,
                geometry=from_shape(simple_poly, srid=4326),
                concentration_percent=concentration + random.uniform(-5, 5), # Add some jitter
                ice_type=ice_type,
                satellite_source='mock_organic',
                processed=True,
                raw_data_url='http://mock-organic-source-global'
            )
            session.add(ice_record)
            features.append(ice_record)
            count += 1
            
    session.commit()
    logger.info(f"Successfully inserted {count} organic ice polygons tailored to Global Arctic geography.")

def main():
    session = get_db_session()
    try:
        generate_mock_data(session)
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()
