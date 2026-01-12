"""
Permafrost stability monitoring model
"""
from sqlalchemy import Column, String, Float, DateTime, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
from datetime import datetime
from ..database import Base


class PermafrostSite(Base):
    __tablename__ = "permafrost_sites"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    site_name = Column(String, nullable=False)
    site_type = Column(String, nullable=False)  # building, mine, infrastructure
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    stability_score = Column(Float, nullable=False)  # 0-100 (higher = more stable)
    temperature_c = Column(Float, nullable=True)
    last_analysis = Column(DateTime, nullable=False)
    alert_level = Column(String, nullable=False, default="stable")  # stable, warning, critical
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<PermafrostSite {self.site_name} - {self.alert_level}>"


class PermafrostData(Base):
    __tablename__ = "permafrost_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    measurement_date = Column(DateTime, nullable=False)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    temperature_celsius = Column(Float, nullable=False)
    depth_meters = Column(Float, nullable=False)
    region_name = Column(String, nullable=False)
    data_source = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<PermafrostData {self.region_name} - {self.temperature_celsius}°C at {self.depth_meters}m>"
