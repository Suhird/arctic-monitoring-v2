/**
 * Permafrost Analysis Map Component
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, Paper, Typography, CircularProgress, Alert, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import api from '../../services/api';

// @ts-ignore
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || '';

interface PermafrostData {
  id: number;
  region_name: string;
  temperature_celsius: number;
  depth_meters: number;
  measurement_date: string;
  data_source: string;
  location: any;
}

interface LocationGroup {
  region: string;
  lat: number;
  lon: number;
  measurements: PermafrostData[];
  avgTemp: number;
}

export default function PermafrostMap() {
  const [locationGroups, setLocationGroups] = useState<LocationGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);

  const getTempColor = (temp: number) => {
    if (temp >= -1) return '#ff5252'; // Very warm (critical)
    if (temp >= -2) return '#ff9800'; // Warm (warning)
    if (temp >= -3) return '#ffeb3b'; // Moderate
    if (temp >= -4) return '#64b5f6'; // Cool
    return '#2196f3'; // Cold (stable)
  };

  const addPermafrostToMap = useCallback((groups: LocationGroup[]) => {
    if (!map.current) return;

    // Remove existing sources/layers
    if (map.current.getSource('permafrost-data')) {
      if (map.current.getLayer('permafrost-points')) {
        map.current.removeLayer('permafrost-points');
      }
      if (map.current.getLayer('permafrost-labels')) {
        map.current.removeLayer('permafrost-labels');
      }
      map.current.removeSource('permafrost-data');
    }

    // Create GeoJSON
    const geojson = {
      type: 'FeatureCollection' as const,
      features: groups.map((group) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Point' as const,
          coordinates: [group.lon, group.lat],
        },
        properties: {
          region: group.region,
          avgTemp: group.avgTemp,
          measurementCount: group.measurements.length,
        },
      })),
    };

    // Add source
    map.current.addSource('permafrost-data', {
      type: 'geojson',
      data: geojson,
    });

    // Determine circle color based on selection
    const circleColor = selectedLocation
      ? [
          'case',
          ['==', ['get', 'region'], selectedLocation],
          '#ffeb3b',
          '#2196f3'
        ]
      : '#2196f3';

    // Add points layer
    map.current.addLayer({
      id: 'permafrost-points',
      type: 'circle',
      source: 'permafrost-data',
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['zoom'],
          2, 8,
          10, 20
        ],
        'circle-color': circleColor as any,
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff',
        'circle-opacity': 0.8,
      },
    });

    // Add labels
    map.current.addLayer({
      id: 'permafrost-labels',
      type: 'symbol',
      source: 'permafrost-data',
      layout: {
        'text-field': ['get', 'region'],
        'text-size': 12,
        'text-offset': [0, 1.5],
        'text-anchor': 'top',
      },
      paint: {
        'text-color': '#ffffff',
        'text-halo-color': '#000000',
        'text-halo-width': 1,
      },
    });

    // Add click handler
    map.current.on('click', 'permafrost-points', (e) => {
      if (!e.features || e.features.length === 0) return;
      const props = e.features[0].properties;
      const group = groups.find(g => g.region === props?.region);

      if (group) {
        setSelectedLocation(props?.region || null);

        const depthMeasurements = group.measurements
          .map(m => `${m.depth_meters}m: ${m.temperature_celsius.toFixed(1)}°C`)
          .join('<br/>');

        new mapboxgl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(`
            <strong>${props?.region}</strong><br/>
            <strong>Avg Temperature:</strong> ${props?.avgTemp?.toFixed(2)}°C<br/>
            <strong>Measurements:</strong> ${props?.measurementCount}<br/>
            <br/>
            <strong>Depth Profiles:</strong><br/>
            ${depthMeasurements}
          `)
          .addTo(map.current!);
      }
    });

    // Hover effects
    map.current.on('mouseenter', 'permafrost-points', () => {
      if (map.current) map.current.getCanvas().style.cursor = 'pointer';
    });
    map.current.on('mouseleave', 'permafrost-points', () => {
      if (map.current) map.current.getCanvas().style.cursor = '';
    });
  }, [selectedLocation]);

  const loadPermafrostData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/v1/permafrost/all');
      const data = response.data;

      // Group by location
      const groups: { [key: string]: LocationGroup } = {};
      data.forEach((item: PermafrostData) => {
        const coords = item.location.coordinates;
        if (!groups[item.region_name]) {
          groups[item.region_name] = {
            region: item.region_name,
            lon: coords[0],
            lat: coords[1],
            measurements: [],
            avgTemp: 0,
          };
        }
        groups[item.region_name].measurements.push(item);
      });

      // Calculate averages
      const groupsArray = Object.values(groups).map(group => {
        const avgTemp = group.measurements.reduce((sum, m) => sum + m.temperature_celsius, 0) / group.measurements.length;
        return { ...group, avgTemp };
      });

      setLocationGroups(groupsArray);
      setError('');
    } catch (err: any) {
      setError('Failed to load permafrost data. Please check if the backend is running.');
      console.error('Error loading permafrost data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPermafrostData();
  }, [loadPermafrostData]);

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-100, 70],
      zoom: 2,
      projection: { name: 'globe' } as any,
    });

    const currentMap = map.current;

    return () => {
      currentMap.remove();
    };
  }, []); // Only initialize once

  useEffect(() => {
    if (locationGroups.length === 0 || !map.current) return;

    const currentMap = map.current;

    if (currentMap.loaded()) {
      addPermafrostToMap(locationGroups);
    } else {
      currentMap.on('load', () => {
        addPermafrostToMap(locationGroups);
      });
    }
  }, [locationGroups, selectedLocation, addPermafrostToMap]);

  const selectedGroup = locationGroups.find(g => g.region === selectedLocation);

  return (
    <Box sx={{ height: 'calc(100vh - 64px)', position: 'relative' }}>
      {loading && (
        <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'rgba(0, 0, 0, 0.7)', zIndex: 1000 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Box sx={{ position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 1001, maxWidth: 600 }}>
          <Alert severity="error">{error}</Alert>
        </Box>
      )}

      <Box ref={mapContainer} sx={{ width: '100%', height: '100%' }} />

      {!loading && !error && (
        <>
          {/* Info Panel */}
          <Paper sx={{ position: 'absolute', top: 16, right: 16, p: 2, minWidth: 320, maxWidth: 400, maxHeight: '80vh', overflow: 'auto', bgcolor: 'rgba(19, 47, 76, 0.95)', zIndex: 999 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Permafrost Monitoring
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              <strong>Monitoring Stations:</strong> {locationGroups.length}
            </Typography>

            {selectedGroup && (
              <Box sx={{ mb: 2, p: 2, bgcolor: 'rgba(33, 150, 243, 0.1)', borderRadius: 1 }}>
                <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                  {selectedGroup.region}
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Depth</TableCell>
                        <TableCell>Temp (°C)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {selectedGroup.measurements
                        .sort((a, b) => a.depth_meters - b.depth_meters)
                        .map((m, idx) => (
                          <TableRow key={idx}>
                            <TableCell>{m.depth_meters}m</TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: getTempColor(m.temperature_celsius) }} />
                                {m.temperature_celsius.toFixed(1)}
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}

            <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
              <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                <strong>Temperature Status:</strong>
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#2196f3' }} />
                <Typography variant="caption">&lt; -4°C (Stable)</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#64b5f6' }} />
                <Typography variant="caption">-4 to -3°C (Cool)</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#ffeb3b' }} />
                <Typography variant="caption">-3 to -2°C (Moderate)</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#ff9800' }} />
                <Typography variant="caption">-2 to -1°C (Warning)</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#ff5252' }} />
                <Typography variant="caption">&gt; -1°C (Critical)</Typography>
              </Box>
            </Box>

            <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
              <Typography variant="caption" color="text.secondary">
                Click on monitoring stations to view depth-temperature profiles
              </Typography>
            </Box>
          </Paper>
        </>
      )}
    </Box>
  );
}
