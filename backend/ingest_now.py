
"""
Script to trigger immediate ingestion of AMSR2 data.
Runs for the last 5 days to ensure we have recent coverage.
"""
import os
import sys
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
print(f"DEBUG: sys.executable = {sys.executable}")
from app.ingestion.bremen import BremenIngester
from app.database import SessionLocal
from app.models.ingestion import IceConcentrationDaily
from sqlalchemy import func

def main():
    print("🚀 Starting manual ingestion trigger (Bremen Source)...")
    
    # Try last 5 days
    today = datetime.utcnow()
    ingester = BremenIngester()
    
    for i in range(5):
        target_date = today - timedelta(days=i)
        print(f"\nProcessing date: {target_date.date()}")
        
        try:
            result = ingester.ingest(target_date)
            count = result.get('records', 0)
            status = result.get('status')
            
            if status == 'success':
                print(f"✅ Successfully ingested {count} polygons for {target_date.date()}")
            elif status == 'skipped' or status == 'failed':
                print(f"⚠️  Ingestion {status}: {result.get('message') or result.get('error')}")
                
                # FALLBACK: Insert synthetic data for development/demo
                print(f"   Generating synthetic demo data for {target_date.date()} so map isn't empty...")
                _insert_synthetic_data(target_date)
                
            else:
                print(f"❌ Unknown status: {status}")
                
        except Exception as e:
            print(f"❌ Error running ingester: {e}")

def _insert_synthetic_data(date_obj):
    """Insert a fake polar cap for visualization testing"""
    db = SessionLocal()
    try:
        # Check if already exists to avoid duplicates
        exists = db.query(IceConcentrationDaily).filter(func.date(IceConcentrationDaily.date) == date_obj.date()).first()
        if exists:
            print("   (Synthetic data already exists)")
            return

        # Create 3 concentric rings of ice (100%, 75%, 40%)
        # Using simple circles (buffered points) around North Pole
        # Note: In EPSG:4326, circles near pole are distorted, but good enough for demo.
        
        # 1. Thick Ice (90-100%) - 80 degrees North
        poly_100 = "POLYGON((-180 80, -90 80, 0 80, 90 80, 180 80, 180 90, -180 90, -180 80))"
        
        # 2. Mid Ice (40-70%) - 70-80 degrees North
        poly_70 = "POLYGON((-180 70, -90 70, 0 70, 90 70, 180 70, 180 80, 90 80, 0 80, -90 80, -180 80, -180 70))"
        
        # 3. Thin Ice (15-40%) - 60-70 degrees North
        poly_30 = "POLYGON((-180 60, -90 60, 0 60, 90 60, 180 60, 180 70, 90 70, 0 70, -90 70, -180 70, -180 60))"

        records = [
            {"conc": 95.0, "geom": poly_100},
            {"conc": 55.0, "geom": poly_70},
            {"conc": 25.0, "geom": poly_30},
        ]

        for r in records:
            ice = IceConcentrationDaily(
                date=date_obj,
                source="SYNTHETIC_DEMO",
                geometry=f"SRID=4326;{r['geom']}",
                concentration=r['conc'],
                resolution_km=12.5
            )
            db.add(ice)
        
        db.commit()
        print("   ✅ Synthetic data inserted.")

    except Exception as e:
        print(f"   ❌ Failed to insert synthetic data: {e}")
        db.rollback()
    finally:
        db.close()

    # Check database status
    db = SessionLocal()
    try:
        count = db.query(IceConcentrationDaily).count()
        latest = db.query(func.max(IceConcentrationDaily.date)).scalar()
        print(f"\n📊 Database Status:")
        print(f"   Total Records: {count}")
        print(f"   Latest Date:   {latest}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
