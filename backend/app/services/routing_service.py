"""
Vessel routing service with A* pathfinding
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import heapq
import math
from global_land_mask import globe
from geoalchemy2.shape import to_shape
from shapely.geometry import Point, box
from ..models.vessel_route import VesselRoute
from ..models.ice_data import IceConcentration
from ..utils.geospatial import calculate_distance_km, linestring_to_wkt
from geoalchemy2.elements import WKTElement

# Vessel capabilities
VESSEL_SPECS = {
    "icebreaker_heavy": {"max_ice_thickness": 3.0, "speed_open": 18, "speed_ice": 6},
    "icebreaker_light": {"max_ice_thickness": 1.5, "speed_open": 20, "speed_ice": 4},
    "merchant_reinforced": {"max_ice_thickness": 0.5, "speed_open": 22, "speed_ice": 2},
}

def calculate_optimal_route(
    db: Session,
    start_coords: Dict[str, float],
    end_coords: Dict[str, float],
    vessel_type: str,
    departure_time: datetime,
    user_id: str = None
) -> Dict[str, Any]:
    """
    Calculate optimal route using A* algorithm on a grid.
    Uses REAL ice data from the database.
    """
    specs = VESSEL_SPECS.get(vessel_type, VESSEL_SPECS["merchant_reinforced"])
    
    # 1. Define Grid Resolution
    lat_step = 0.5
    lon_step = 1.0
    
    start_node = (round(start_coords["lat"], 1), round(start_coords["lon"], 1))
    end_node = (round(end_coords["lat"], 1), round(end_coords["lon"], 1))
    
    # 2. Build Ice Risk Map (In-Memory Raster)
    # We fetch ice polygons for the relevant area and rasterize them to our grid
    # This avoids thousands of DB queries during A*
    
    min_lat = min(start_node[0], end_node[0]) - 10
    max_lat = max(start_node[0], end_node[0]) + 10
    # Handle longitude wrap-around simply for MVP (fetching a wide adequate strip)
    min_lon = -180
    max_lon = 180
    
    ice_map = _build_ice_risk_map(db, min_lat, max_lat, min_lon, max_lon, lat_step, lon_step)

    def get_node_cost(lat, lon):
        """
        Returns (traversable: bool, cost: float, risk: float, speed: float, ice_desc: str)
        """
        # 1. Check Land
        if globe.is_land(lat, lon):
             return False, float('inf'), 100, 0, "Land"

        # 2. Get Ice Conditions from Map
        # Round to nearest grid point key
        grid_lat = round(lat / lat_step) * lat_step
        grid_lon = round(lon / lon_step) * lon_step
        
        # Simple key lookup
        ice_info = ice_map.get((grid_lat, grid_lon), {"conc": 0, "thick": 0})
        
        ice_concentration = ice_info["conc"]
        # If thickness is missing, approximate it from concentration (MVP heuristic)
        # 100% conc ~ 2.0m, 10% ~ 0.2m
        ice_thickness = ice_info["thick"]
        if ice_thickness == 0 and ice_concentration > 0:
             ice_thickness = (ice_concentration / 100.0) * 2.0
        
        # 3. Check Vessel Capability
        if ice_thickness > specs["max_ice_thickness"]:
            return False, float('inf'), 100, 0, f"Too thick ({ice_thickness:.1f}m > {specs['max_ice_thickness']}m)"

        # 4. Calculate traversing cost
        speed = specs["speed_open"]
        if ice_concentration > 0:
            speed_penalty_factor = (ice_concentration / 100)
            speed = specs["speed_open"] * (1 - speed_penalty_factor) + specs["speed_ice"] * speed_penalty_factor
        
        speed = max(0.1, speed)
        
        # Cost is hours to cross + risk penalty
        risk = (ice_concentration / 100) * (ice_thickness / specs["max_ice_thickness"]) * 100
        
        desc = f"Open Water"
        if ice_concentration > 5:
             desc = f"Ice: {ice_concentration:.0f}% ({ice_thickness:.1f}m)"
             
        return True, (1/speed), risk, speed, desc

    # 3. A* Algorithm
    frontier = []
    heapq.heappush(frontier, (0, start_node))
    came_from = {}
    cost_so_far = {}
    
    node_metadata = {} 
    
    came_from[start_node] = None
    cost_so_far[start_node] = 0
    node_metadata[start_node] = {"risk": 0, "speed": specs["speed_open"], "desc": "Start"}
    
    found = False
    max_iter = 15000 
    iter_count = 0
    
    final_node = end_node

    while frontier and iter_count < max_iter:
        iter_count += 1
        current_cost, current = heapq.heappop(frontier)
        
        # Relaxed goal check
        if abs(current[0] - end_node[0]) < lat_step and abs(current[1] - end_node[1]) < lon_step:
            final_node = current
            found = True
            break
            
        for dlat in [-lat_step, 0, lat_step]:
            for dlon in [-lon_step, 0, lon_step]:
                if dlat == 0 and dlon == 0: continue
                
                neighbor = (round(current[0] + dlat, 2), round(current[1] + dlon, 2))
                
                # Basic bounds
                if neighbor[0] > 90 or neighbor[0] < 50: continue
                if neighbor[1] <= -180: neighbor = (neighbor[0], 180.0)
                if neighbor[1] > 180: neighbor = (neighbor[0], -179.0)
                
                # Distance (km)
                lat_rad = math.radians(neighbor[0])
                dist_km = math.sqrt((dlat * 111.32)**2 + (dlon * 111.32 * math.cos(lat_rad))**2)
                
                # Check traversability
                traversable, time_cost_factor, risk, speed, desc = get_node_cost(neighbor[0], neighbor[1])
                
                if not traversable: continue
                
                # Heuristic Weighting
                edge_cost = (dist_km / speed) + (risk * 0.05) 
                
                new_cost = cost_so_far[current] + edge_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + _heuristic(neighbor, end_node)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
                    node_metadata[neighbor] = {"risk": risk, "speed": speed, "desc": desc}

    # Reconstruct path
    path = []
    if found:
        curr = final_node
        while curr != start_node:
            meta = node_metadata.get(curr, {})
            path.append({
                "lat": curr[0], 
                "lon": curr[1],
                "speed_kts": round(meta.get("speed", 0), 1),
                "ice_risk": round(meta.get("risk", 0), 1),
                "condition": meta.get("desc", "")
            })
            curr = came_from[curr]
        path.append({"lat": start_node[0], "lon": start_node[1], "condition": "Start"})
        path.reverse()
    else:
        # Fallback
        path = _generate_waypoints(start_coords, end_coords, 10)
        for p in path: p.update({"condition": "Fallback - No safe path found", "ice_risk": 0})

    # Calculate totals
    total_dist = 0
    total_time = 0
    max_risk = 0
    
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        d = calculate_distance_km(p1["lon"], p1["lat"], p2["lon"], p2["lat"])
        total_dist += d
        speed = p1.get("speed_kts", 10)
        speed_kmh = speed * 1.852
        if speed_kmh > 0:
            total_time += d / speed_kmh
        max_risk = max(max_risk, p1.get("ice_risk", 0))

    # Save to DB
    linestring_coords = [[p["lon"], p["lat"]] for p in path]
    path_wkt = linestring_to_wkt(linestring_coords)
    
    route = VesselRoute(
        route_name=f"Route {start_coords['lat']},{start_coords['lon']} to {end_coords['lat']},{end_coords['lon']}",
        start_point=WKTElement(f"POINT({start_coords['lon']} {start_coords['lat']})", srid=4326),
        end_point=WKTElement(f"POINT({end_coords['lon']} {end_coords['lat']})", srid=4326),
        route_geometry=WKTElement(path_wkt, srid=4326),
        ice_risk_score=max_risk,
        distance_km=total_dist,
        estimated_duration_hours=total_time,
        user_id=user_id,
        vessel_id=vessel_type
    )
    
    try:
        db.add(route)
        db.commit()
    except Exception as e:
        print(f"Error saving route: {e}")
        db.rollback()

    return {
        "route_id": route.id if route.id else 0,
        "path": {
            "type": "LineString",
            "coordinates": linestring_coords
        },
        "ice_risk_score": max_risk,
        "estimated_duration_hours": total_time,
        "total_distance_km": total_dist,
        "waypoints": path[::5] # Return fewer waypoints for UI performance, but keep full path in geometry
    }

def _build_ice_risk_map(db: Session, min_lat, max_lat, min_lon, max_lon, lat_step, lon_step) -> Dict[Tuple[float, float], Dict]:
    """
    Fetch relevant ice polygons and map them to grid points max concentration.
    """
    ice_map = {}
    
    # 1. Get recent ice data (last 48 hours to be safe)
    cutoff = datetime.utcnow() - timedelta(hours=48)
    
    # We select polygons that MIGHT overlap our area
    # Note: Global wrap around makes bbox query tricky, so we query loosely
    features = db.query(IceConcentration).filter(
        IceConcentration.timestamp >= cutoff
    ).all()
    
    if not features:
        # No real data found? Return empty map (Open Water assumption)
        # OR fallback to simulation for demo if DB is empty
        # For now, return empty (open water)
        return ice_map

    # 2. Rasterize
    # This naive O(N*M) approach is slow if many features.
    # Optimized: Iterate over grid points and check containment? Still slow.
    # Best: Iterate over Features, find their bbox, map to grid indices.
    
    for feature in features:
        conc = feature.concentration_percent
        poly = to_shape(feature.geometry)
        bounds = poly.bounds # (minx, miny, maxx, maxy)
        
        # Determine grid range containing this polygon
        f_min_lat = max(min_lat, bounds[1])
        f_max_lat = min(max_lat, bounds[3])
        f_min_lon = max(min_lon, bounds[0])
        f_max_lon = min(max_lon, bounds[2])
        
        # Loop steps
        idx_lat_start = int(f_min_lat / lat_step)
        idx_lat_end = int(f_max_lat / lat_step) + 1
        idx_lon_start = int(f_min_lon / lon_step) 
        idx_lon_end = int(f_max_lon / lon_step) + 1
        
        for i_lat in range(idx_lat_start, idx_lat_end):
            for i_lon in range(idx_lon_start, idx_lon_end):
                lat = i_lat * lat_step
                lon = i_lon * lon_step
                
                # Check point in poly
                if poly.contains(Point(lon, lat)):
                     key = (round(lat, 2), round(lon, 2))
                     # Keep max concentration at this point
                     if key not in ice_map or conc > ice_map[key]["conc"]:
                         ice_map[key] = {"conc": conc, "thick": 0} # Thickness could be added if in DB

    return ice_map

def _heuristic(a, b):
    # Great circle distance approximation
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in km
    return c * r

def _generate_waypoints(start, end, num_points):
    waypoints = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = start["lat"] + t * (end["lat"] - start["lat"])
        lon = start["lon"] + t * (end["lon"] - start["lon"])
        waypoints.append({"lat": lat, "lon": lon})
    return waypoints

