
import os
import requests
from io import BytesIO
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

class MaxarDataFetcher:
    """
    Fetcher for Maxar SecureWatch / WMS / WMTS API.
    Requires MAXAR_API_KEY environment variable.
    """
    def __init__(self):
        self.api_key = os.getenv("MAXAR_API_KEY")
        # Base URL for Maxar WMS (Example URL, may vary by subscription)
        # Often: https://securewatch.maxar.com/mapservice/wmsaccess
        self.base_url = "https://securewatch.maxar.com/mapservice/wmsaccess"

    def fetch_tile_image(self, bbox: list, width=256, height=256) -> Optional[bytes]:
        """
        Fetch a WMS tile from Maxar.
        Bbox: [min_x, min_y, max_x, max_y] in EPSG:3857 (Web Mercator)
        """
        if not self.api_key:
            return self._generate_mock_tile(width, height, "Maxar API Key Missing")

        # Construct WMS Request
        # Note: Maxar WMS usually requires 'connectId' or Basic Auth
        params = {
            "connectId": self.api_key,
            "service": "WMS",
            "request": "GetMap",
            "version": "1.3.0",
            "layers": "DigitalGlobe:Imagery", # Common layer name
            "crs": "EPSG:3857",
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "width": width,
            "height": height,
            "format": "image/png",
            "transparent": "true"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                if "image" in response.headers.get("Content-Type", ""):
                    return response.content
                else:
                    return self._generate_mock_tile(width, height, "Maxar Auth Failed")
            else:
                return self._generate_mock_tile(width, height, f"Maxar Error: {response.status_code}")
        except Exception as e:
            print(f"Maxar Fetch Error: {e}")
            return self._generate_mock_tile(width, height, "Connection Error")

    def _generate_mock_tile(self, width, height, message="Maxar Premium"):
        """Generates a placeholder tile for demo/fallback."""
        img = Image.new('RGBA', (width, height), (20, 20, 20, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw a grid pattern to look technical
        for i in range(0, width, 20):
            draw.line([(i, 0), (i, height)], fill=(50, 50, 50), width=1)
            draw.line([(0, i), (width, i)], fill=(50, 50, 50), width=1)
            
        # Draw "Premium" text visualization (Gold Cross)
        draw.line([(0, 0), (width, height)], fill=(255, 215, 0), width=3)
        draw.line([(width, 0), (0, height)], fill=(255, 215, 0), width=3)
        draw.rectangle([10, 10, width-10, height-10], outline=(255, 215, 0), width=3)
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

_fetcher = None

def get_maxar_fetcher():
    global _fetcher
    if _fetcher is None:
        _fetcher = MaxarDataFetcher()
    return _fetcher
