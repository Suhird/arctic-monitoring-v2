"""
Sea Ice Thickness data model (from Altimetry)
"""
from sqlalchemy import Column, Float, DateTime, BigInteger, String
from geoalchemy2 import Geometry
from datetime import datetime
from ..database import Base

class IceThickness(Base):
    __tablename__ = "ice_thickness"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    thickness_meters = Column(Float, nullable=False)
    uncertainty_meters = Column(Float, nullable=True)
    source_satellite = Column(String, nullable=False)  # CryoSat-2, ICESat-2, SMOS
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<IceThickness {self.id} - {self.thickness_meters}m>"
