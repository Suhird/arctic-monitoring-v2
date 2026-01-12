"""
Ice data schemas for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class IceConcentrationResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[Dict[str, Any]]


class IceTimeSeriesPoint(BaseModel):
    timestamp: datetime
    concentration_percent: float
    ice_type: Optional[str]


class IceTimeSeriesResponse(BaseModel):
    bbox: List[float]
    data: List[IceTimeSeriesPoint]


class BBoxParams(BaseModel):
    min_lon: float = Field(..., ge=-180, le=180)
    min_lat: float = Field(..., ge=-90, le=90)
    max_lon: float = Field(..., ge=-180, le=180)
    max_lat: float = Field(..., ge=-90, le=90)

    def to_wkt(self) -> str:
        """Convert bbox to WKT POLYGON"""
        return f"POLYGON(({self.min_lon} {self.min_lat}, {self.max_lon} {self.min_lat}, {self.max_lon} {self.max_lat}, {self.min_lon} {self.max_lat}, {self.min_lon} {self.min_lat}))"


class HistoricalPatternsRequest(BaseModel):
    bbox: BBoxParams
    start_year: int
    end_year: int


class HistoricalPatternsResponse(BaseModel):
    seasonal_patterns: Dict[str, Any]
    trends: Dict[str, Any]
    statistics: Dict[str, float]
