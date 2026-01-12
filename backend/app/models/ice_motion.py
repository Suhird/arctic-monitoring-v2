"""
Sea Ice Motion data model
"""
from sqlalchemy import Column, Float, DateTime, BigInteger, String
from geoalchemy2 import Geometry
from datetime import datetime
from ..database import Base

class IceMotion(Base):
    __tablename__ = "ice_motion"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    u_vector_cm_sec = Column(Float, nullable=False)  # Velocity Eastward
    v_vector_cm_sec = Column(Float, nullable=False)  # Velocity Northward
    source = Column(String, nullable=False) # Sentinel-1, AMSR2
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<IceMotion {self.id} - ({self.u_vector_cm_sec}, {self.v_vector_cm_sec})>"
