"""
Ice concentration data model with PostGIS geometry
"""
from sqlalchemy import Column, String, Float, Boolean, DateTime, BigInteger
from geoalchemy2 import Geometry
from datetime import datetime
from ..database import Base


class IceConcentration(Base):
    __tablename__ = "ice_concentration"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    geometry = Column(Geometry(geometry_type='POLYGON', srid=4326), nullable=False)
    concentration_percent = Column(Float, nullable=False)  # 0-100
    ice_type = Column(String, nullable=True)  # first_year, multi_year, pack_ice
    satellite_source = Column(String, nullable=False)  # sentinel1, sentinel2, radarsat, etc.
    raw_data_url = Column(String, nullable=True)
    processed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<IceConcentration {self.id} - {self.concentration_percent}% at {self.timestamp}>"


class SatelliteImagery(Base):
    __tablename__ = "satellite_imagery"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    acquisition_time = Column(DateTime, nullable=False, index=True)
    satellite_source = Column(String, nullable=False)
    bounds = Column(Geometry(geometry_type='POLYGON', srid=4326), nullable=False)
    file_path = Column(String, nullable=False)
    cloud_cover_percent = Column(Float, nullable=True)
    processed = Column(Boolean, default=False, nullable=False)
    image_metadata = Column(String, nullable=True)  # JSON as string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<SatelliteImagery {self.id} - {self.satellite_source} at {self.acquisition_time}>"
