
from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, Index
from datetime import datetime
from app.database import Base

class SatelliteTile(Base):
    __tablename__ = "satellite_tiles"

    x = Column(Integer, primary_key=True)
    y = Column(Integer, primary_key=True)
    z = Column(Integer, primary_key=True)
    
    # Store the PNG bytes
    image = Column(LargeBinary, nullable=False)
    
    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # We might want to store which time range this represents
    # but for now, this table represents the "Latest 7 Day Mosaic"
