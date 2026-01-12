"""
University of Bremen AMSR2 Ice Concentration Ingester
Source: https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250/
"""
from typing import Any, List, Dict, Optional
from datetime import datetime
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import numpy as np

from .base import BaseIngester
from ..database import SessionLocal
from ..models.ingestion import IceConcentrationDaily

class BremenIngester(BaseIngester):
    """
    Ingests daily AMSR2 sea ice concentration from University of Bremen.
    Resolution: 6.25km (n6250)
    Format: GeoTIFF
    """
    
    def __init__(self):
        super().__init__(source_name="BREMEN_ASI", data_type="concentration")
        self.base_url = "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250"
        self.current_target_date = None

    def fetch_data(self, target_date: datetime) -> Optional[bytes]:
        """
        Download GeoTIFF for the target date.
        URL Format: .../2026/jan/Arctic/asi-AMSR2-n6250-20260101-v5.4.tif
        """
        self.current_target_date = target_date
        
        year = target_date.strftime("%Y")
        month_abbr = target_date.strftime("%b").lower() # jan, feb...
        date_str = target_date.strftime("%Y%m%d")
        
        # Construct URL
        # Note: v5.4 is current version as seen in 2026 directory
        filename = f"asi-AMSR2-n6250-{date_str}-v5.4.tif"
        url = f"{self.base_url}/{year}/{month_abbr}/Arctic/{filename}"
        
        print(f"[BREMEN] Requesting: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                print(f"[BREMEN] Downloaded {len(response.content)} bytes")
                return response.content
            elif response.status_code == 404:
                print(f"[BREMEN] Data not found for {target_date.date()} (404)")
                return None
            else:
                print(f"[BREMEN] Failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"[BREMEN] Error fetching: {e}")
            return None

    def process_data(self, raw_data: bytes) -> List[Dict]:
        """
        Vectorize the GeoTIFF. 
        Bremen Data: 0-100 = Concentration, >100 = Flags (Land, Missing)
        """
        if not raw_data:
            return []
            
        records = []
        
        try:
            with MemoryFile(raw_data) as memfile:
                with memfile.open() as src:
                    ice = src.read(1)
                    transform = src.transform

                    # LOG BOUNDS
                    try:
                        from rasterio.warp import transform as transform_coords
                        width = src.width
                        height = src.height
                        # Grid corners
                        tl = src.transform * (0, 0)
                        tr = src.transform * (width, 0)
                        br = src.transform * (width, height)
                        bl = src.transform * (0, height)
                        xs = [tl[0], tr[0], br[0], bl[0]]
                        ys = [tl[1], tr[1], br[1], bl[1]]
                        # Reproject to WGS84
                        lons, lats = transform_coords(src.crs, {'init': 'epsg:4326'}, xs, ys)
                        print(f"VVVVV BREMEN BOUNDS VVVVV")
                        print(f"[[{lons[0]}, {lats[0]}], [{lons[1]}, {lats[1]}], [{lons[2]}, {lats[2]}], [{lons[3]}, {lats[3]}]]")
                        print(f"^^^^^ BREMEN BOUNDS ^^^^^")
                    except Exception as e:
                        print(f"Error calculating bounds: {e}")
                    
                    # Bremen Coding:
                    # 0-100: Concentration %
                    # 119: Land
                    # 120: Missing
                    
                    # Define ranges to simplify polygons (reduce DB load)
                    ranges = [
                        (15, 40),   # Low
                        (40, 70),   # Medium
                        (70, 85),   # High
                        (85, 101)   # Very High
                    ]
                    
                    for min_val, max_val in ranges:
                        # Mask for valid ice in this range
                        mask = (ice >= min_val) & (ice < max_val)
                        
                        if not mask.any():
                            continue
                            
                        # Extract shapes
                        geoms = []
                        for geom, val in shapes(mask.astype(np.uint8), transform=transform):
                            if val == 1: # Start logic often returns value of mask (1)
                                try:
                                    s = shape(geom)
                                    if s.is_valid:
                                        geoms.append(s)
                                    else:
                                        print(f"Invalid geometry skipped")
                                        geoms.append(s.buffer(0)) # Try fixing
                                except Exception:
                                    pass

                        if geoms:
                            try:
                                # Union to create MultiPolygon
                                unified = unary_union(geoms)
                                
                                # Simplify to reduce size (0.01 degrees ~ 1km)
                                simplified = unified.simplify(0.01, preserve_topology=True)
                                
                                avg_conc = (min_val + max_val) / 2
                                
                                # Reproject logic
                                from pyproj import Transformer
                                from shapely.ops import transform as shapely_transform

                                # Create transformer (Source CRS -> EPSG:4326)
                                # Rasterio 'src.crs' usually holds the projection, usually Polar Stereographic
                                # We'll define it based on src.crs
                                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform
                                
                                # Transform to WGS84
                                reprojected = shapely_transform(transformer, simplified)
                                
                                records.append({
                                    "geometry": reprojected.wkt,
                                    "concentration": avg_conc,
                                    "resolution_km": 6.25
                                })
                            except Exception as e:
                                print(f"Error merging geometries: {e}")

        except Exception as e:
            print(f"[BREMEN] Processing error: {e}")
            return []
            
        return records

    def store_data(self, processed_data: List[Dict]) -> int:
        """Store in IceConcentrationDaily"""
        if not processed_data:
            return 0
            
        db = SessionLocal()
        count = 0
        try:
            # First, check if we already have data for this date/source to avoid cleanup mess
            # Ideally we might delete old data for this day before inserting
            # but usually ingest shouldn't duplicate if logical checks exist.
            # Here we just append.
            
            for record in processed_data:
                entry = IceConcentrationDaily(
                    date=self.current_target_date or datetime.utcnow(),
                    source=self.source_name,
                    geometry=f"SRID=4326;{record['geometry']}",
                    concentration=record['concentration'],
                    resolution_km=record['resolution_km']
                )
                db.add(entry)
                count += 1
            
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[BREMEN] Storage error: {e}")
            count = 0
        finally:
            db.close()
            
        return count
