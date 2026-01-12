"""
Base Ingester Abstract Class
All data ingesters inherit from this to ensure consistency.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models.ingestion import DataIngestionLog


class BaseIngester(ABC):
    """Abstract base class for all data ingesters"""
    
    def __init__(self, source_name: str, data_type: str):
        self.source_name = source_name
        self.data_type = data_type
        self.db: Optional[Session] = None
        self.log_id: Optional[int] = None
        
    def _start_logging(self, target_date: datetime) -> int:
        """Create ingestion log entry"""
        self.db = SessionLocal()
        log = DataIngestionLog(
            source=self.source_name,
            data_type=self.data_type,
            target_date=target_date,
            status="running",
            started_at=datetime.utcnow()
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        self.log_id = log.id
        return log.id
    
    def _complete_logging(self, status: str, records: int = 0, error: str = None):
        """Update ingestion log with completion status"""
        if not self.db or not self.log_id:
            return
            
        log = self.db.query(DataIngestionLog).filter_by(id=self.log_id).first()
        if log:
            log.status = status
            log.records_ingested = records
            log.error_message = error
            log.completed_at = datetime.utcnow()
            if log.started_at:
                log.duration_seconds = (log.completed_at - log.started_at).total_seconds()
            self.db.commit()
        
        self.db.close()
        self.db = None
        self.log_id = None
    
    @abstractmethod
    def fetch_data(self, target_date: datetime) -> Any:
        """
        Fetch raw data from source for the given date.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def process_data(self, raw_data: Any) -> List[Dict]:
        """
        Process raw data into standardized format.
        Must be implemented by subclasses.
        Returns list of records ready for database insertion.
        """
        pass
    
    @abstractmethod
    def store_data(self, processed_data: List[Dict]) -> int:
        """
        Store processed data in database.
        Must be implemented by subclasses.
        Returns number of records stored.
        """
        pass
    
    def ingest(self, target_date: datetime) -> Dict[str, Any]:
        """
        Main ingestion workflow.
        Can be overridden if custom workflow needed.
        """
        self._start_logging(target_date)
        
        try:
            # Fetch
            print(f"[{self.source_name}] Fetching data for {target_date.date()}...")
            raw_data = self.fetch_data(target_date)
            
            if not raw_data:
                self._complete_logging("failed", error="No data available")
                return {"status": "failed", "message": "No data available"}
            
            # Process
            print(f"[{self.source_name}] Processing data...")
            processed_data = self.process_data(raw_data)
            
            # Store
            print(f"[{self.source_name}] Storing data...")
            records_count = self.store_data(processed_data)
            
            self._complete_logging("success", records=records_count)
            print(f"[{self.source_name}] ✓ Ingested {records_count} records")
            
            return {
                "status": "success",
                "records": records_count,
                "date": target_date.date().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Ingestion failed: {str(e)}"
            print(f"[{self.source_name}] ✗ {error_msg}")
            self._complete_logging("failed", error=error_msg)
            return {"status": "failed", "message": error_msg}
    
    def ingest_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Ingest data for a date range.
        """
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            result = self.ingest(current_date)
            results.append(result)
            current_date += timedelta(days=1)
        
        return results
