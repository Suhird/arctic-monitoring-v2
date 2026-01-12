"""
Sea Ice Extent data model
"""
from sqlalchemy import Column, Float, DateTime, BigInteger, String
from datetime import datetime
from ..database import Base

class SeaIceExtent(Base):
    __tablename__ = "sea_ice_extent"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    extent_sq_km = Column(Float, nullable=False)
    area_sq_km = Column(Float, nullable=True)  # Actual ice area (excluding open water in pixels)
    hemisphere = Column(String, default="north", nullable=False)
    source = Column(String, default="NSIDC", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<SeaIceExtent {self.date.date()} - {self.extent_sq_km} km2>"
