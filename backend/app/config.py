"""
Configuration management for Arctic Ice Platform
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Arctic Ice Monitoring Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str
    DB_ECHO: bool = False

    # Redis
    REDIS_URL: str = "redis://redis:6379"

    # JWT Authentication
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:80"  # Comma-separated string

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # Satellite Data APIs
    # Legacy SciHub (Deprecated)
    SENTINEL_USERNAME: str = ""
    SENTINEL_PASSWORD: str = ""
    
    # Copernicus Data Space Ecosystem (New)
    CDSE_CLIENT_ID: str = ""
    CDSE_CLIENT_SECRET: str = ""

    RADARSAT_API_KEY: str = ""
    PLANET_API_KEY: str = ""
    MAXAR_API_KEY: str = ""
    
    # NSIDC Credentials (Earthdata) - DEPRECATED (Removed)

    # Mapbox
    MAPBOX_TOKEN: str = ""

    # ML Models
    # Use absolute path relative to this file to avoid CWD ambiguity
    # Note: os.path.dirname(__file__) is backend/app
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml/models")
    
    # Cache
    TIFF_CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/cache/tiff_files_cache")
    
    ICE_CLASSIFIER_MODEL: str = "ice_classifier_resnet50.pth"
    ICE_PREDICTOR_MODEL: str = "ice_movement_lstm.pth"
    CHANGE_DETECTOR_MODEL: str = "change_detector.pth"

    # Cache TTL (seconds)
    CACHE_ICE_CURRENT: int = 300  # 5 minutes
    CACHE_PREDICTION_7DAY: int = 3600  # 1 hour
    CACHE_ROUTE: int = 1800  # 30 minutes

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # Data Storage
    SATELLITE_DATA_PATH: str = "/data/satellite"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env
    )


settings = Settings()
