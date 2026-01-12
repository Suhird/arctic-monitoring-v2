"""
Vessel route recommendations model
"""
from sqlalchemy import Column, String, Float, DateTime, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
from datetime import datetime
from ..database import Base


class VesselRoute(Base):
    __tablename__ = "vessel_routes"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    vessel_id = Column(String, nullable=True, default="demo-vessel")
    route_name = Column(String, nullable=True)
    start_point = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    end_point = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    route_geometry = Column(Geometry(geometry_type='LINESTRING', srid=4326), nullable=False)
    distance_km = Column(Float, nullable=True)
    estimated_duration_hours = Column(Float, nullable=True)
    ice_risk_score = Column(Float, nullable=True)  # 0-10 risk score
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)

    def __repr__(self):
        return f"<VesselRoute {self.route_name} - Risk: {self.ice_risk_score}>"
