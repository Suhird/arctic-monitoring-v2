import sys
import os
from datetime import datetime, timedelta
import logging

# Ensure backend directory is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.ingestion.bremen import BremenIngester

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HistoricalIngest")

def ingest_last_30_days():
    ingester = BremenIngester()
    
    # Start from yesterday (T-1) and go back 30 days
    today = datetime.now()
    
    for i in range(1, 31):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")
        
        logger.info(f"Processing date: {date_str}...")
        
        try:
            # 1. Fetch
            raw_data = ingester.fetch_data(target_date)
            if not raw_data:
                logger.warning(f"No data found for {date_str}, skipping.")
                continue
                
            # 2. Process
            records = ingester.process_data(raw_data)
            logger.info(f"  Processed {len(records)} records for {date_str}")
            
            # 3. Store
            count = ingester.store_data(records)
            logger.info(f"  ✅ Stored {count} records for {date_str}")
            
        except Exception as e:
            logger.error(f"Failed to ingest {date_str}: {e}")

if __name__ == "__main__":
    logger.info("Starting historical data ingestion (Last 30 days)...")
    ingest_last_30_days()
    logger.info("Ingestion complete.")
