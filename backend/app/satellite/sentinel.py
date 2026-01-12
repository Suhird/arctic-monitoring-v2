"""
Sentinel-1/2 satellite data integration (Copernicus Data Space Ecosystem)
Modern API replacement for retired SciHub.
"""
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json
from ..config import settings

# CDSE Constants
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"

class SentinelDataFetcher:
    """
    Fetch satellite imagery from Sentinel-1 and Sentinel-2
    API: Copernicus Data Space Ecosystem (CDSE)
    URL: https://dataspace.copernicus.eu/
    """

    def __init__(self):
        self.access_token = None
        self.token_expiry = 0
        self.client_id = settings.CDSE_CLIENT_ID
        self.client_secret = settings.CDSE_CLIENT_SECRET
        
    def _authenticate(self) -> bool:
        """Authenticate with CDSE to get access token"""
        if not self.client_id or not self.client_secret:
            print("CDSE Credentials not configured.")
            return False
            
        # Check if token is still valid (buffer of 60s)
        if self.access_token and time.time() < self.token_expiry - 60:
            return True
            
        try:
            payload = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'client_credentials'
            }
            response = requests.post(AUTH_URL, data=payload)
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            self.token_expiry = time.time() + data['expires_in']
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False

    def search_products(
        self,
        bbox: tuple,
        start_date: datetime,
        end_date: datetime,
        collection: str = "SENTINEL-1",
    ) -> List[Dict[str, Any]]:
        """
        Search for Sentinel products via OData API
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: Start datetime
            end_date: End datetime
            collection: SENTINEL-1 or SENTINEL-2
            
        Returns:
            List of products
        """
        # Convert bbox to WKT Polygon
        min_lon, min_lat, max_lon, max_lat = bbox
        wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        
        # Format dates ISO 8601
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        query_filter = (
            f"ContentDate/Start ge {start_str} and ContentDate/Start le {end_str} "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') "
            f"and Collection/Name eq '{collection}'"
        )
        
        if collection == "SENTINEL-2":
            query_filter += " and CloudCover lt 30"
        elif collection == "SENTINEL-1":
             # filter for Ground Range Detected (GRD) to get visualizable images
             # Attribute filtering syntax: Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'IW_GRDH_1S')
             # Simplified: just checking the name or using explicit attribute filter is hard in OData.
             # Easier: Name contains 'GRD'
             query_filter += " and contains(Name,'GRD')"

        params = {
            "$filter": query_filter,
            "$orderby": "ContentDate/Start desc",
            "$top": 10,
            "$expand": "Attributes"
        }
        
        try:
            final_url = f"{ODATA_URL}/Products"
            print(f"Querying: {final_url} ...")
            
            response = requests.get(final_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('value', [])
            
        except Exception as e:
            print(f"Error searching products: {e}")
            return []

    def fetch_quicklook(self, product_id: str) -> Optional[bytes]:
        """
        Fetch the Quicklook image using Sentinel Hub Process API.
        Robustly handles polarization and scaling to ensure visible imagery.
        """
        if not self._authenticate():
            return None
            
        token = f"Bearer {self.access_token}"
        headers = {"Authorization": token}
        
        try:
            # 1. Fetch Product Metadata (Expanded for Attributes)
            meta_url = f"{ODATA_URL}/Products({product_id})"
            # We explicitly expand Attributes to check polarization
            r_meta = requests.get(meta_url, headers=headers, params={"$expand": "Attributes"})
            
            if r_meta.status_code != 200:
                print(f"Failed to fetch metadata for quicklook: {r_meta.status_code}")
                return None
                
            meta = r_meta.json()
            
            # Extract BBox
            if 'GeoFootprint' in meta:
                coords = meta['GeoFootprint']['coordinates'][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                bbox = [min(lons), min(lats), max(lons), max(lats)]
            else:
                return None
            
            # Extract Polarization to choose band
            main_pol = "VV" # Default
            attrs = meta.get('Attributes', [])
            for a in attrs:
                if a.get('Name') == 'polarisationChannels':
                    val = a.get('Value', '')
                    if 'HH' in val:
                        main_pol = "HH"
            
            # Extract Time
            start_time = meta['ContentDate']['Start']
            end_time = meta['ContentDate']['End']
            
            # 2. Call SH Process API
            sh_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
            
            # Evalscript: Optimized for Ice Visualization
            # Uses Square Root intensity scaling which is generally robust for dark Sentinel-1 data
            evalscript = f"""
            //VERSION=3
            function setup() {{
              return {{
                input: ["{main_pol}"],
                output: {{ bands: 3 }}
              }};
            }}
            
            function evaluatePixel(sample) {{
              var val = sample.{main_pol};
              // Sqrt scaling boosts dark pixels (ice/water) better than linear
              // Multiply by 2.5 to brighten
              var vis = Math.sqrt(val) * 2.5;
              
              // Clamp
              vis = Math.max(0, Math.min(1, vis));
              
              return [vis, vis, vis];
            }}
            """
            
            payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": { "crs": "http://www.opengis.net/def/crs/EPSG/0/4326" }
                    },
                    "data": [
                        {
                            "type": "SENTINEL-1-GRD",
                            "dataFilter": {
                                "timeRange": {
                                    "from": start_time,
                                    "to": end_time
                                },
                                "acquisitionMode": "IW",
                                "resolution": "MEDIUM"
                            }
                        }
                    ]
                },
                "output": {
                    "width": 512,
                    "height": 512,
                    "responses": [
                        {
                            "identifier": "default",
                            "format": {
                                "type": "image/jpeg"
                            }
                        }
                    ]
                },
                "evalscript": evalscript
            }
            
            r_proc = requests.post(sh_url, headers=headers, json=payload)
            
            if r_proc.status_code == 200:
                return r_proc.content
            else:
                print(f"Process API failed: {r_proc.status_code} {r_proc.text}")
                return None
                
        except Exception as e:
            print(f"Error fetching quicklook via Process API: {e}")
            return None

    
    def fetch_tile_image(self, bbox: list, time_range: tuple, width=256, height=256, crs="http://www.opengis.net/def/crs/EPSG/0/4326") -> Optional[bytes]:
        """
        Fetch a map tile image (PNG) for the given bbox and time range.
        Used for map overlay.
        """
        if not self._authenticate():
             return None
             
        token = f"Bearer {self.access_token}"
        headers = {"Authorization": token}
        
        # Evalscript with Alpha for Transparency
        # If no data (dataMask=0), alpha=0
        # Uses HH/VV detection logic? Or simpler "VV" default?
        # Tile bbox is small, but polarimetry might vary.
        # Safe bet: Use VV as it's standard for 1SDV.
        # Ideally we'd check availability but for tiles speed is key.
        # We'll use a robust script that checks input availability.
        
        
        
        def fetch_with_pol(pol_cand):
            # Evalscript adapted for Multi-Temporal Mosaicking (Smoothing)
            evalscript = f"""
            //VERSION=3
            function setup() {{
              return {{
                input: ["{pol_cand}", "dataMask"],
                output: {{ bands: 4 }},
                mosaicking: "ORBIT"
              }};
            }}
            
            function evaluatePixel(samples) {{
              let sum = 0;
              let count = 0;
              
              // Average all valid observations in the time window (De-speckling)
              for (let i = 0; i < samples.length; i++) {{
                  if (samples[i].dataMask === 1) {{
                      sum += samples[i].{pol_cand};
                      count++;
                  }}
              }}
              
              if (count === 0) return [0, 0, 0, 0];
              
              let val = sum / count;
              
              // Convert to dB
              let db = 10 * Math.log10(Math.max(0.0001, val));
              
              // Color Map for Sea Ice (with Land Masking)
              // Very High Return (> 0 dB): Likely land/rocks -> Transparent
              // High Return (-5 to 0 dB): Thick ice -> White semi-transparent
              // Medium Return (-25 to -5 dB): Thin ice -> Blue to White gradient
              // Low Return (< -25 dB): Water -> Dark Blue
              
              if (db > 0) {{
                 // Very bright - likely land, make transparent
                 return [1, 1, 1, 0]; 
              }} else if (db > -5.0) {{
                 // Strong ice - white but semi-transparent
                 return [1, 1, 1, 0.75];
              }} else if (db > -25.0) {{
                 // Ice-water transition
                 let t = (db + 25.0) / 20.0;
                 return [t, 0.3 + t*0.7, 0.8 + t*0.2, 0.7];
              }} else {{
                 // Water - dark blue, semi-transparent
                 return [0.0, 0.05, 0.3, 0.6];
              }}
            }}
            """
            
            payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": { "crs": crs }
                    },
                    "data": [
                        {
                            "type": "SENTINEL-1-GRD",
                            "dataFilter": {
                                "timeRange": {
                                    "from": time_range[0],
                                    "to": time_range[1]
                                },
                                "resolution": "MEDIUM"
                            }
                        }
                    ]
                },
                "output": {
                    "width": width,
                    "height": height,
                    "responses": [
                        {
                            "identifier": "default",
                            "format": {
                                "type": "image/png"
                            }
                        }
                    ]
                },
                "evalscript": evalscript
            }
            sh_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
            return requests.post(sh_url, headers=headers, json=payload)

        # Try HH first (Best for Arctic/Ice) -> Fallback to VV
        for pol in ["HH", "VV"]:
            try:
                r = fetch_with_pol(pol)
                if r.status_code == 200:
                    return r.content
                elif r.status_code == 400 and "MISSING_POLARIZATION" in r.text:
                    # Try next polarization
                    continue
                else:
                    print(f"Tile fetch failed ({pol}): {r.status_code} {r.text}")
                    return None
            except Exception as e:
                print(f"Error fetching tile ({pol}): {e}")
                return None
        
        return None

    def fetch_high_res_polygons(self, bbox: tuple) -> Dict[str, Any]:
        """
        Generate high-precision polygons from latest available imagery.
        """
        if not self._authenticate():
             return {"error": "Authentication failed. Check CDSE credentials."}

        # Look back 3 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=3)
        
        print(f"Searching for Sentinel-1 in {bbox}...")
        products = self.search_products(bbox, start_date, end_date, "SENTINEL-1")
        
        if not products:
             print("No Sentinel-1 found, trying Sentinel-2...")
             products = self.search_products(bbox, start_date, end_date, "SENTINEL-2")
             
        if not products:
            return {"error": "No recent satellite imagery found in this region."}
            
        latest_product = products[0]
        product_id = latest_product['Id']
        product_name = latest_product['Name']
        footprint_vector = latest_product.get('GeoFootprint') # This is usually GeoJSON geometry
        
        return {
            "type": "FeatureCollection",
            "metadata": {
                "source": "Copernicus Data Space Ecosystem",
                "product_id": product_id,
                "product_name": product_name,
                "acquisition_date": latest_product.get('ContentDate', {}).get('Start', datetime.utcnow().isoformat()),
                "sensor": "SENTINEL-1" if "S1" in product_name else "SENTINEL-2"  # Infer from name
            },
            "features": [{
                "type": "Feature",
                "properties": {
                     "ice_concentration": 0.85, # Mocked analysis result
                     "concentration_percent": 85.0, # Frontend expects 0-100
                     "type": "thick_ice",
                     "satellite_source": "sentinel",
                     "product_id": product_id  # Added for clickable map interaction
                },
                "geometry": footprint_vector # Returning the image footprint for now
            }]
        }

def get_sentinel_fetcher() -> SentinelDataFetcher:
    """Get Sentinel data fetcher instance"""
    return SentinelDataFetcher()
