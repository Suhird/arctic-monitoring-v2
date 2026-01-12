"""
Ice prediction model for forecasting ice movement
"""
from sqlalchemy import Column, String, Float, DateTime, BigInteger, Integer
from geoalchemy2 import Geometry
from datetime import datetime
from ..database import Base


class IcePrediction(Base):
    __tablename__ = "ice_predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    prediction_date = Column(DateTime, nullable=False, index=True)  # Date being predicted
    forecast_days = Column(Integer, nullable=False)  # Number of days ahead (1-7)
    geometry = Column(Geometry(geometry_type='POLYGON', srid=4326), nullable=False)
    predicted_concentration = Column(Float, nullable=False)  # 0-100
    confidence_score = Column(Float, nullable=True)  # 0-1
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<IcePrediction {self.id} - Day {self.forecast_days}>"
