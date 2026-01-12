"""
Satellite data integration package
"""
from .sentinel import get_sentinel_fetcher
from .radarsat import get_radarsat_fetcher
from .planet_labs import get_planet_labs_fetcher
from .maxar import get_maxar_fetcher
from .processor import get_image_processor

__all__ = [
    "get_sentinel_fetcher",
    "get_radarsat_fetcher",
    "get_planet_labs_fetcher",
    "get_maxar_fetcher",
    "get_image_processor",
]
