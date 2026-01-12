
"""
Mock Data Generator
Generates simulated satellite data for demo purposes when real APIs are unavailable.
"""
import random
import math
from datetime import datetime
from typing import Dict, Any, List

def generate_mock_ice_data(bbox: tuple = (-180, 60, 180, 90)) -> Dict[str, Any]:
    """
    Generate simulated ice concentration data (GeoJSON FeatureCollection).
    Creates a realistic-looking seasonal ice cap.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    features = []
    
    # Grid resolution (degrees)
    step = 2.0 
    
    # Seasonality
    now = datetime.utcnow()
    day_of_year = now.timetuple().tm_yday
    # Minimum extent in Sept (day ~260), Max in March (day ~75)
    # Cosine factor: 1.0 (Max ice) to 0.0 (Min ice)
    season_factor = (math.cos((day_of_year - 75) / 365.0 * 2 * math.pi) + 1) / 2
    
    # Ice edge latitude varies by season: 
    # Winter: ~60N, Summer: ~75N
    ice_edge_lat = 75 - (season_factor * 15) 

    for lon in range(int(min_lon), int(max_lon), int(step)):
        for lat in range(int(min_lat), int(max_lat), int(step)):
            
            # Distance from pole (roughly)
            # Simple approximation: darker/thicker near 90N
            
            # Base concentration based on latitude
            if lat < ice_edge_lat:
                continue # Open water
                
            dist_from_edge = lat - ice_edge_lat
            
            # Concentration ramps up from edge to pole
            concentration = min(100, (dist_from_edge / 10.0) * 100)
            
            # Add noise/fractures
            noise = random.uniform(-15, 15)
            concentration = max(0, min(100, concentration + noise))
            
            if concentration < 15:
                continue

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "concentration_percent": round(concentration, 1),
                    "ice_type": "multi_year" if concentration > 80 else "first_year",
                    "satellite_source": "Simulation",
                    "timestamp": now.isoformat()
                }
            }
            features.append(feature)
            
    return {
        "type": "FeatureCollection",
        "features": features
    }
