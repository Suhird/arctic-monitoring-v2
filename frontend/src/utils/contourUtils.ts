
import { contours } from 'd3-contour';
import { scaleLinear } from 'd3-scale';

type Point = {
  lat: number;
  lon: number;
  value: number;
};

/**
 * Converts GeoJSON Point collection to Contour MultiPolygons
 * @param geoJson GeoJSON FeatureCollection of Points
 * @param width Grid width (longitude resolution)
 * @param height Grid height (latitude resolution)
 */
export function generateIceContours(geoJson: any, width: number = 200, height: number = 100) {
  if (!geoJson || !geoJson.features || geoJson.features.length === 0) return null;

  // 1. Extract points
  const points: Point[] = geoJson.features.map((f: any) => {
    let lon, lat;
    if (f.geometry.type === 'Point') {
        lon = f.geometry.coordinates[0];
        lat = f.geometry.coordinates[1];
    } else if (f.geometry.type === 'Polygon') {
        // Calculate centroid of the polygon (outer ring)
        const ring = f.geometry.coordinates[0];
        let sumLon = 0, sumLat = 0;
        ring.forEach((coord: number[]) => {
            sumLon += coord[0];
            sumLat += coord[1];
        });
        lon = sumLon / ring.length;
        lat = sumLat / ring.length;
    } else {
        return null;
    }

    return {
        lon,
        lat,
        value: f.properties.concentration_percent || 0
    };
  }).filter((p: any) => p !== null);
  console.log(`[ContourUtils] Extracted ${points.length} points.`);

  // 2. Define Bounds (Arctic)
  const MIN_LON = -180, MAX_LON = 180;
  const MIN_LAT = 60, MAX_LAT = 90;

  // 3. Create Grid
  const values = new Float64Array(width * height).fill(0);
  const counts = new Uint16Array(width * height).fill(0);

  // Helper to map values to grid index
  const getIndex = (lon: number, lat: number) => {
    // Normalize lon to 0..width
    const x = Math.floor(((lon - MIN_LON) / (MAX_LON - MIN_LON)) * width);
    // Normalize lat to 0..height (Lat increases upwards usually, but d3-contour creates image coords)
    // We map 90N to y=0 and 60N to y=height to match image coords top-down
    const y = Math.floor(((MAX_LAT - lat) / (MAX_LAT - MIN_LAT)) * height);
    
    if (x >= 0 && x < width && y >= 0 && y < height) {
      return y * width + x;
    }
    return -1;
  };

  // 4. Bin values (Simple averaging)
  let binsFilled = 0;
  points.forEach(p => {
    const idx = getIndex(p.lon, p.lat);
    if (idx !== -1) {
      if (counts[idx] === 0) binsFilled++;
      values[idx] += p.value;
      counts[idx] += 1;
    }
  });
  console.log(`[ContourUtils] Filled ${binsFilled} bins out of ${width*height}.`);

  // Average and smooth
  for (let i = 0; i < values.length; i++) {
    if (counts[i] > 0) values[i] /= counts[i];
  }

  // 5. Generate Contours
  // Thresholds: 10%, 20% ... 100%
  const thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
  const contourGenerator = contours()
    .size([width, height])
    .thresholds(thresholds)
    .smooth(true);

  const rawContours = contourGenerator(Array.from(values));
  console.log(`[ContourUtils] Generated ${rawContours.length} raw contour sets.`);

  // 6. Transform back to GeoJSON
  // d3-contour returns coordinates in [x, y] pixels. We need to map back to [lon, lat]
  const transformCoordinates = (ring: number[][]) => {
    return ring.map(([x, y]) => {
      const lon = MIN_LON + (x / width) * (MAX_LON - MIN_LON);
      const lat = MAX_LAT - (y / height) * (MAX_LAT - MIN_LAT);
      return [lon, lat];
    });
  };

  const features = rawContours.map(c => {
    if (!c.coordinates || c.coordinates.length === 0) return null;
    
    return {
      type: 'Feature',
      properties: {
        value: c.value, // Threshold value (e.g., 10, 20...)
        fillColor: getColorForConcentration(c.value) // Assign color helper
      },
      geometry: {
        type: 'MultiPolygon',
        coordinates: c.coordinates.map((polygon: any) => 
          polygon.map((ring: any) => transformCoordinates(ring))
        )
      }
    };
  }).filter(f => f !== null);

  console.log(`[ContourUtils] Returning ${features.length} separate features/bands.`);

  return {
    type: 'FeatureCollection',
    features: features
  };
}

function getColorForConcentration(value: number): string {
    if (value < 15) return 'rgba(0,0,0,0)'; // Transparent
    if (value < 30) return '#4fc3f7'; // Light Blue
    if (value < 50) return '#29b6f6'; 
    if (value < 70) return '#039be5'; 
    if (value < 90) return '#0277bd';
    return '#e1f5fe'; // Near white/solid for deepest ice
}
