import os
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session
from geoalchemy2.elements import WKTElement

# Add backend to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.ice_data import SatelliteImagery

# Configuration
CACHE_DIR = Path("data/cache/bremen")
BASE_URL = "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250"
DAYS_TO_FETCH = 30

def download_image(date: datetime, style: str = "visual") -> bool:
    """
    Download image for a specific date and style.
    Returns True if downloaded or already exists, False if failed.
    """
    yyyy = date.strftime("%Y")
    mm = date.strftime("%b").lower()
    yyyymmdd = date.strftime("%Y%m%d")
    
    filename = f"{date.strftime('%Y-%m-%d')}_{style}.png"
    filepath = CACHE_DIR / filename
    
    # ensure directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        print(f"   Using cached: {filename}")
        return True

    upstream_filename = f"asi-AMSR2-n6250-{yyyymmdd}-v5.4_{style}.png"
    url = f"{BASE_URL}/{yyyy}/{mm}/Arctic/{upstream_filename}"
    
    print(f"   Downloading: {url}")
    
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"   ✅ Saved to {filepath}")
            return True
        elif response.status_code == 404:
            print(f"   ⚠️  Not found (404)")
            return False
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def register_in_db(db: Session, date: datetime, style: str):
    """Register the image in the SatelliteImagery table"""
    filename = f"{date.strftime('%Y-%m-%d')}_{style}.png"
    filepath = str(CACHE_DIR / filename) # relative path preferred? keeping relative for now or absolute based on implementation
    
    # Check if exists
    exists = db.query(SatelliteImagery).filter(
        SatelliteImagery.acquisition_time == date,
        SatelliteImagery.satellite_source == f"Bremen_{style}"
    ).first()
    
    if exists:
        return

    # Create entry
    # Using a generic full Arctic bounding box for Bremen
    # Polygon covering approx Arctic: (-180 60, -180 90, 180 90, 180 60, -180 60)
    bbox_wkt = "POLYGON((-180 60, -180 90, 180 90, 180 60, -180 60))"
    
    entry = SatelliteImagery(
        acquisition_time=date,
        satellite_source=f"Bremen_{style}",
        bounds=WKTElement(bbox_wkt, srid=4326),
        file_path=filepath,
        processed=True,
        image_metadata='{"provider": "University of Bremen", "type": "Visual"}'
    )
    db.add(entry)
    db.commit()
    print(f"   📝 Registered in DB")

def main():
    print(f"🚀 Starting Historical Download for last {DAYS_TO_FETCH} days...")
    print(f"   Output Directory: {CACHE_DIR.resolve()}\n")
    
    db = SessionLocal()
    
    try:
        today = datetime.utcnow()
        success_count = 0
        
        for i in range(1, DAYS_TO_FETCH + 1):
            target_date = today - timedelta(days=i)
            print(f"[{target_date.date()}] Processing...")
            
            if download_image(target_date):
                try:
                    register_in_db(db, target_date, "visual")
                    success_count += 1
                except Exception as db_err:
                    print(f"   ❌ DB Error: {db_err}")
                    db.rollback()
            
            print("-" * 40)
            
        print(f"\n✨ Completed! Successfully cached {success_count}/{DAYS_TO_FETCH} days.")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
