import rasterio
from rasterio.warp import transform_bounds
import requests
import io
import datetime

# URL for a sample Bremen GeoTIFF (e.g., yesterday or recently known good date)
# Using fixed date 2026-01-09 as known available from previous steps
DATE_STR = "20260109"
YEAR = "2026"
MONTH_ABBR = "jan"
URL = f"https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250/{YEAR}/{MONTH_ABBR}/Arctic/asi-AMSR2-n6250-{DATE_STR}-v5.4.tif"

def get_bounds():
    print(f"Downloading sample TIF from: {URL}")
    response = requests.get(URL)
    if response.status_code != 200:
        print(f"Error downloading: {response.status_code}")
        return

    with rasterio.open(io.BytesIO(response.content)) as src:
        print(f"Source CRS: {src.crs}")
        print(f"Source Bounds: {src.bounds}")
        
        # Calculate WGS84 bounds (EPSG:4326)
        # transform_bounds(src_crs, dst_crs, left, bottom, right, top)
        wgs84_bounds = transform_bounds(src.crs, {'init': 'epsg:4326'}, *src.bounds)
        print(f"WGS84 Bounds (min_lon, min_lat, max_lon, max_lat): {wgs84_bounds}")
        
        # Mapbox 'image' source requires 4 coordinates: [top-left, top-right, bottom-right, bottom-left]
        # But we need to handle the reprojection carefully. 
        # Ideally, we get the lat/lon of the 4 corners of the raster.
        
        from rasterio.warp import transform
        
        height = src.height
        width = src.width
        
        # Corners in source pixels: (0,0), (width, 0), (width, height), (0, height)
        # Convert to source CRS coordinates
        # src.transform * (col, row) -> (x, y)
        tl = src.transform * (0, 0)
        tr = src.transform * (width, 0)
        br = src.transform * (width, height)
        bl = src.transform * (0, height)
        
        xs = [tl[0], tr[0], br[0], bl[0]]
        ys = [tl[1], tr[1], br[1], bl[1]]
        
        lons, lats = transform(src.crs, {'init': 'epsg:4326'}, xs, ys)
        
        mapbox_coords = [
            [lons[0], lats[0]], # Top-Left
            [lons[1], lats[1]], # Top-Right
            [lons[2], lats[2]], # Bottom-Right
            [lons[3], lats[3]]  # Bottom-Left
        ]
        
        print("\nMapbox Image Coordinates:")
        print(mapbox_coords)

if __name__ == "__main__":
    get_bounds()
