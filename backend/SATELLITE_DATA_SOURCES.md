# Arctic Ice Satellite Data Sources

This document describes the free satellite data sources integrated into the Arctic Ice Monitoring Platform.

## 📡 Data Sources

### 1. NSIDC Sea Ice Index (Currently Integrated)

**Provider:** National Snow and Ice Data Center (NSIDC)
**Status:** ✅ Fully Integrated
**Data Access:** Free, No API Key Required

#### What it provides:
- Daily Arctic sea ice concentration data
- Updated daily with 1-day lag
- Spatial resolution: 25km x 25km grid
- Data format: GeoTIFF
- Coverage: November 1978 - Present
- Hemispheres: North and South

#### Data Access:
- **Direct URL:** `https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/geotiff/[YEAR]/`
- **Format:** `N_[YYYYMMDD]_concentration_v3.0.tif`
- **Example:** `https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/geotiff/2025/N_20251230_concentration_v3.0.tif`

#### How to Use in the App:
1. Navigate to the Ice Concentration Map
2. Click **"Use Real Satellite Data"** button
3. The app will automatically download and process the latest NSIDC data

#### Technical Details:
- **Sensor:** AMSR2 (Advanced Microwave Scanning Radiometer 2)
- **Satellite:** GCOM-W1 (Japan Aerospace Exploration Agency)
- **Algorithm:** NASA Team Algorithm
- **Processing:** Automated daily updates

**Documentation:** [NSIDC Sea Ice Index](https://nsidc.org/data/g02135/versions/3)

---

### 2. ESA Copernicus Sentinel-1 (Available for Integration)

**Provider:** European Space Agency (ESA)
**Status:** 🔄 Ready for Integration
**Data Access:** Free with Registration

#### What it provides:
- High-resolution SAR (Synthetic Aperture Radar) imagery
- All-weather, day-night imaging capability
- Spatial resolution: 5m - 40m
- Revisit time: 6 days (with Sentinel-1A and 1B)
- Real-time ice edge detection
- Ice type classification

#### Data Access:
- **Platform:** Copernicus Data Space Ecosystem
- **URL:** https://dataspace.copernicus.eu/
- **APIs Available:**
  - STAC API (SpatioTemporal Asset Catalog)
  - openEO API
  - Sentinel Hub API
  - OData API

#### Registration Process:
1. Visit https://dataspace.copernicus.eu/
2. Create a free account
3. Generate API credentials
4. Access via programmatic APIs or web interface

#### How to Integrate:
```python
# Example using Sentinel Hub API
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection

config = SHConfig()
config.sh_client_id = 'YOUR_CLIENT_ID'
config.sh_client_secret = 'YOUR_CLIENT_SECRET'

# Request SAR imagery for Arctic region
request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL1_IW,
            time_interval=('2025-12-30', '2025-12-31'),
        )
    ],
    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
    bbox=bbox,
    size=[512, 512],
    config=config
)
```

**Documentation:** [Copernicus Data Space](https://documentation.dataspace.copernicus.eu/)

---

### 3. NASA Worldview (Available for Integration)

**Provider:** NASA EOSDIS
**Status:** 🔄 Ready for Integration
**Data Access:** Free, No Registration Required

#### What it provides:
- Near real-time satellite imagery
- Multiple sensor combinations
- Corrected Reflectance imagery
- Sea ice concentration and extent
- Visual RGB imagery of Arctic

#### Data Access:
- **Web Interface:** https://worldview.earthdata.nasa.gov/
- **API:** GIBS (Global Imagery Browse Services)
- **URL:** https://gibs.earthdata.nasa.gov/
- **Format:** WMTS (Web Map Tile Service)

#### Available Layers for Arctic:
- MODIS Corrected Reflectance
- VIIRS Corrected Reflectance
- Sea Ice Concentration (AMSR2)
- Sea Ice Extent
- Sea Surface Temperature

#### How to Integrate:
```python
# Example GIBS WMTS request
base_url = "https://gibs.earthdata.nasa.gov/wmts/epsg3413/best/wmts.cgi"
params = {
    'SERVICE': 'WMTS',
    'REQUEST': 'GetTile',
    'VERSION': '1.0.0',
    'LAYER': 'AMSR2_Sea_Ice_Concentration_12km',
    'STYLE': 'default',
    'TILEMATRIXSET': 'EPSG3413_250m',
    'TILEMATRIX': '4',
    'TILEROW': '0',
    'TILECOL': '0',
    'FORMAT': 'image/png',
    'TIME': '2025-12-30'
}
```

**Documentation:** [NASA GIBS](https://wiki.earthdata.nasa.gov/display/GIBS)

---

### 4. NOAA CoastWatch (Available for Integration)

**Provider:** NOAA
**Status:** 🔄 Ready for Integration
**Data Access:** Free via ERDDAP

#### What it provides:
- Sea ice concentration
- Sea surface temperature
- Ocean color
- Near real-time data
- NetCDF and GeoTIFF formats

#### Data Access:
- **Platform:** ERDDAP (Environmental Research Division's Data Access Program)
- **URL:** https://coastwatch.pfeg.noaa.gov/erddap/
- **Protocol:** RESTful API with multiple output formats

#### How to Integrate:
```python
# Example ERDDAP request
base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
dataset = "jplMURSST41"  # Multi-scale Ultra-high Resolution SST
params = {
    'latitude': '(60):1:(90)',
    'longitude': '(-180):1:(180)',
    'time': '(2025-12-30T12:00:00Z)'
}
url = f"{base_url}/{dataset}.geotiff?{params}"
```

**Documentation:** [NOAA CoastWatch](https://coastwatch.noaa.gov/)

---

### 5. Canadian Ice Service (Available for Integration)

**Provider:** Environment and Climate Change Canada
**Status:** 🔄 Ready for Integration
**Data Access:** Free, No Registration

#### What it provides:
- Daily ice charts
- Ice concentration analysis
- Ice type and stage of development
- Detailed Arctic coverage
- SIGRID-3 format (XML/Shapefile)

#### Data Access:
- **Web Service:** https://ice-glaces.ec.gc.ca/
- **Format:** ESRI Shapefile, GeoTIFF, GeoJSON
- **Coverage:** Canadian Arctic and sub-Arctic waters

#### How to Integrate:
```python
# Example: Download ice chart shapefile
url = "https://ice-glaces.ec.gc.ca/IceGraph/page.xhtml?lang=en"
# Access via WMS/WFS services
```

**Documentation:** [Canadian Ice Service](https://ice-glaces.ec.gc.ca/)

---

## 🔧 Implementation Status

### Currently Implemented
✅ **NSIDC Sea Ice Index**
- Automatic daily data download
- GeoTIFF processing
- Real-time map visualization
- Accessible via UI toggle button

### Next Steps for Integration
1. **Sentinel-1 SAR Data**
   - Register for Copernicus account
   - Implement Sentinel Hub API
   - Add high-resolution ice edge detection

2. **NASA Worldview Integration**
   - Add GIBS WMTS layer support
   - Implement visual imagery overlay
   - Add multi-sensor comparison

3. **Data Fusion**
   - Combine multiple sources for enhanced accuracy
   - Cross-validate data from different sensors
   - Provide confidence scores

---

## 📊 Data Comparison

| Source | Resolution | Update Frequency | Coverage | Best Use Case |
|--------|-----------|------------------|----------|---------------|
| **NSIDC Sea Ice Index** | 25 km | Daily | Global | Trend analysis, extent monitoring |
| **Sentinel-1 SAR** | 5-40 m | 6 days | On-demand | Ice edge detection, ship routing |
| **NASA Worldview** | 250-1000 m | Daily | Global | Visual inspection, RGB imagery |
| **NOAA CoastWatch** | 1-4 km | Daily | Polar regions | SST, ocean monitoring |
| **Canadian Ice Service** | Varies | Daily | Canadian Arctic | Detailed regional analysis |

---

## 🚀 How to Access Real Data in the App

1. **Start the Application**
   ```bash
   # Backend
   cd backend
   uvicorn app.main:app --reload

   # Frontend
   cd frontend
   npm start
   ```

2. **Navigate to Ice Concentration Map**
   - Open http://localhost:3000
   - Click on "Ice Concentration Map" in sidebar

3. **Toggle Real Satellite Data**
   - Click **"Use Real Satellite Data"** button
   - Wait for download and processing (first time may take 30-60 seconds)
   - View real NSIDC satellite data on the map

4. **Refresh Data**
   - Click **"Refresh Data"** to get latest satellite imagery
   - Data updates daily at ~18:00 UTC

---

## 📝 Data Citation

When using data from these sources, please cite:

**NSIDC:**
```
Fetterer, F., K. Knowles, W. N. Meier, M. Savoie, and A. K. Windnagel. 2017,
updated daily. Sea Ice Index, Version 3. Boulder, Colorado USA.
NSIDC: National Snow and Ice Data Center.
https://doi.org/10.7265/N5K072F8
```

**Sentinel-1:**
```
ESA Copernicus Sentinel-1 mission. European Space Agency.
https://sentinel.esa.int/web/sentinel/missions/sentinel-1
```

---

## 🔗 Useful Links

- [NSIDC Homepage](https://nsidc.org/home)
- [Copernicus Data Space](https://dataspace.copernicus.eu/)
- [NASA Earthdata](https://www.earthdata.nasa.gov/)
- [NOAA CoastWatch](https://coastwatch.noaa.gov/)
- [Canadian Ice Service](https://ice-glaces.ec.gc.ca/)

---

**Last Updated:** December 31, 2025
