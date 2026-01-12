"""
Data Ingestion Models
Tracks daily ingestion of various satellite data sources.
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float
from geoalchemy2 import Geometry
from datetime import datetime
from ..database import Base


class IceConcentrationDaily(Base):
    """Daily ice concentration data from AMSR2"""
    __tablename__ = "ice_concentrations_daily"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), default="AMSR2")  # AMSR2, AMSR-E, etc.
    geometry = Column(Geometry('MULTIPOLYGON', srid=4326), nullable=False)
    concentration = Column(Float, nullable=False)  # 0-100%
    resolution_km = Column(Float, default=6.25)
    created_at = Column(DateTime, default=datetime.utcnow)


class IceDriftVector(Base):
    """Ice drift/movement vectors"""
    __tablename__ = "ice_drift_vectors"
    
    id = Column(Integer, primary_key=True, index=True)
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), default="OSI_SAF")  # OSI_SAF, NSIDC, Sentinel1
    start_point = Column(Geometry('POINT', srid=4326), nullable=False)
    end_point = Column(Geometry('POINT', srid=4326), nullable=False)
    velocity_x = Column(Float)  # cm/s or km/day
    velocity_y = Column(Float)
    velocity_magnitude = Column(Float)
    resolution_km = Column(Float, default=62.5)
    created_at = Column(DateTime, default=datetime.utcnow)


class IceExtentDaily(Base):
    """Daily ice extent from MASIE and other sources"""
    __tablename__ = "ice_extent_daily"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), default="MASIE")
    region = Column(String(100))  # Arctic, Antarctic, specific regions
    geometry = Column(Geometry('MULTIPOLYGON', srid=4326), nullable=False)
    area_km2 = Column(Float)
    perimeter_km = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataIngestionLog(Base):
    """Log of all data ingestion runs"""
    __tablename__ = "data_ingestion_log"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False, index=True)  # AMSR2, MASIE, OSI_SAF, etc.
    data_type = Column(String(50), nullable=False)  # concentration, extent, drift
    target_date = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), nullable=False)  # success, failed, partial
    records_ingested = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
