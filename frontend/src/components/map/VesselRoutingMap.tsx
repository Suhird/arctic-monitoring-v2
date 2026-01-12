/**
 * Vessel Routing Map Component
 * Interactive routing tool integrated into dashboard
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Box, Paper, Typography, CircularProgress, Alert, List, ListItem, ListItemText, Chip,
  TextField, MenuItem, Button, Grid, IconButton, Divider
} from '@mui/material';
import { Navigation, AccessTime, Explore, Warning, ArrowBack, Layers } from '@mui/icons-material';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { iceService } from '../../services/iceService';
import { BoundingBox, RouteResponse } from '../../types';

// @ts-ignore
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || '';

const DEFAULT_BBOX: BoundingBox = { min_lon: -180, min_lat: 60, max_lon: 180, max_lat: 90 };

const PORTS = {
  "Barrow, AK": { lat: 71.2906, lon: -156.7887 },
  "Nome, AK": { lat: 64.5011, lon: -165.4064 },
  "Prudhoe Bay, AK": { lat: 70.2552, lon: -148.3372 },
  "Tuktoyaktuk, CA": { lat: 69.4454, lon: -133.0291 },
  "Nuuk, GL": { lat: 64.1814, lon: -51.7231 },
  "Murmansk, RU": { lat: 68.9585, lon: 33.0827 },
  "Tromso, NO": { lat: 69.6492, lon: 18.9553 },
  "Reykjavik, IS": { lat: 64.1466, lon: -21.9426 }
};

export default function VesselRoutingMap() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Routing State
  const [startPort, setStartPort] = useState('Nome, AK');
  const [endPort, setEndPort] = useState('Barrow, AK');
  const [vesselType, setVesselType] = useState('icebreaker_light');
  const [routeData, setRouteData] = useState<RouteResponse | null>(null);
  
  // Data State
  const [iceData, setIceData] = useState<any>(null);
  const [loading, setLoading] = useState(false); // Calculation loading
  const [dataLoading, setDataLoading] = useState(true); // Initial data loading
  const [error, setError] = useState('');
  const [useRealData, setUseRealData] = useState(true);

  // Fetch Ice Data
  useEffect(() => {
    const fetchIceData = async () => {
        setDataLoading(true);
        try {
            const data = await iceService.getCurrentIceData(DEFAULT_BBOX, useRealData); 
            setIceData(data);
        } catch (e) {
            console.error("Failed to load ice context", e);
        } finally {
            setDataLoading(false);
        }
    };
    fetchIceData();
  }, [useRealData]);

  // Initial Map Setup
  useEffect(() => {
    if (!mapContainer.current || map.current) return;
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-100, 75],
      zoom: 2.5,
      projection: { name: 'globe' } as any,
    });

    map.current.on('load', () => {
      setMapLoaded(true);
    });

    return () => map.current?.remove();
  }, []);

  // Update Ice Layer (Restored to Dotted Heatmap Style)
  useEffect(() => {
    if (mapLoaded && iceData && map.current) {
        // Cleanup
        if (map.current.getSource('ice-data')) {
            if (map.current.getLayer('ice-heatmap')) map.current.removeLayer('ice-heatmap');
            if (map.current.getLayer('ice-points')) map.current.removeLayer('ice-points');
            map.current.removeSource('ice-data');
        }

        map.current.addSource('ice-data', {
            type: 'geojson',
            data: iceData
        });

        const beforeId = map.current.getLayer('route-line') ? 'route-line' : undefined;

        // 1. Heatmap Layer (Broad view)
        map.current.addLayer({
            id: 'ice-heatmap',
            type: 'heatmap',
            source: 'ice-data',
            maxzoom: 9, // Switch to dots when zoomed in
            paint: {
                'heatmap-weight': ['interpolate', ['linear'], ['get', 'concentration_percent'], 0, 0, 100, 1],
                'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
                'heatmap-color': [
                    'interpolate', ['linear'], ['heatmap-density'],
                    0, 'rgba(0, 0, 255, 0)',
                    0.2, 'rgba(65, 105, 225, 0.5)',
                    1, 'rgba(255, 255, 255, 0.9)'
                ],
                'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 5, 9, 20],
                'heatmap-opacity': 0.7
            }
        }, beforeId);

        // 2. Circle Layer (Detailed view - "Dotted")
        map.current.addLayer({
            id: 'ice-points',
            type: 'circle',
            source: 'ice-data',
            minzoom: 8,
            paint: {
                'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 2, 16, 10],
                'circle-color': [
                    'interpolate', ['linear'], ['get', 'concentration_percent'],
                    0, 'rgba(0,0,0,0)',
                    20, '#4169e1', 
                    100, '#ffffff'
                ],
                'circle-opacity': 0.8
            }
        }, beforeId);
    }
  }, [mapLoaded, iceData]);

  // Route Calculation
  const calculateRoute = async () => {
    if (!map.current || !mapLoaded) return;
    setLoading(true);
    setError('');
    setRouteData(null);

    try {
      // @ts-ignore
      const start = PORTS[startPort];
      // @ts-ignore
      const end = PORTS[endPort];

      const result = await iceService.calculateRoute(start, end, vesselType);
      setRouteData(result);
      drawRoute(result);
      
    } catch (err: any) {
      console.error(err);
      setError('Route calculation failed. ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const drawRoute = (route: RouteResponse) => {
    if (!map.current) return;

    // Cleanup previous route
    if (map.current.getSource('route')) {
      if (map.current.getLayer('route-line')) map.current.removeLayer('route-line');
      map.current.removeSource('route');
    }

    map.current.addSource('route', {
      type: 'geojson',
      data: {
        type: 'Feature',
        properties: {},
        geometry: route.path
      }
    });

    map.current.addLayer({
      id: 'route-line',
      type: 'line',
      source: 'route',
      layout: { 'line-join': 'round', 'line-cap': 'round' },
      paint: {
        'line-color': route.ice_risk_score > 50 ? '#ff1744' : '#00e676',
        'line-width': 5
      }
    });

    // Fit bounds
    const coords = route.path.coordinates;
    const bounds = new mapboxgl.LngLatBounds(coords[0] as any, coords[0] as any);
    coords.forEach(c => bounds.extend(c as [number, number]));
    map.current.fitBounds(bounds, { padding: 100 });
  };

  return (
    <Box sx={{ height: 'calc(100vh - 64px)', position: 'relative' }}>
        
      {/* Map Element */}
      <Box ref={mapContainer} sx={{ width: '100%', height: '100%' }} />

      {/* Floating Control Panel */}
      <Paper sx={{ 
        position: 'absolute', top: 16, right: 16, p: 3, 
        width: 380, maxHeight: '90vh', overflowY: 'auto',
        bgcolor: 'rgba(19, 47, 76, 0.95)', zIndex: 999 
      }}>
        <Typography variant="h6" color="primary" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Explore /> Smart Vessel Routing
        </Typography>

        <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12}>
              <TextField
                select fullWidth label="Departure" value={startPort}
                onChange={(e) => setStartPort(e.target.value)} size="small"
                sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}
              >
                {Object.keys(PORTS).map(k => <MenuItem key={k} value={k}>{k}</MenuItem>)}
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                select fullWidth label="Destination" value={endPort}
                onChange={(e) => setEndPort(e.target.value)} size="small"
                sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}
              >
                {Object.keys(PORTS).map(k => <MenuItem key={k} value={k}>{k}</MenuItem>)}
              </TextField>
            </Grid>
            <Grid item xs={12}>
               <TextField
                select fullWidth label="Vessel Class" value={vesselType}
                onChange={(e) => setVesselType(e.target.value)} size="small"
                sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}
              >
                <MenuItem value="icebreaker_heavy">Polar Class 1 (Heavy)</MenuItem>
                <MenuItem value="icebreaker_light">Polar Class 4 (Light)</MenuItem>
                <MenuItem value="merchant_reinforced">Merchant Reinforced</MenuItem>
              </TextField>
            </Grid>
        </Grid>

        <Button
            fullWidth variant={useRealData ? "contained" : "outlined"}
            size="small" onClick={() => setUseRealData(!useRealData)}
            sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.3)', color: 'white', bgcolor: useRealData ? 'primary.main' : 'transparent' }}
        >
            {useRealData ? "Source: Real Satellite" : "Source: Simulation"}
        </Button>

        <Button 
            fullWidth variant="contained" size="large"
            onClick={calculateRoute} disabled={!mapLoaded || loading}
            startIcon={loading ? <CircularProgress size={20} color="inherit"/> : <Navigation />}
            sx={{ mb: 3 }}
        >
            {loading ? "Optimizing Route..." : "Calculate Route"}
        </Button>

        {error && (
            <Alert severity="error" sx={{ mb: 2, bgcolor: 'rgba(255, 23, 68, 0.1)', color: '#ff1744' }}>{error}</Alert>
        )}

        {routeData && (
            <Box>
                <Divider sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.1)' }} />
                <Typography variant="subtitle2" color="gray" gutterBottom>ROUTE METRICS</Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                    <Grid item xs={6}>
                        <Paper sx={{ p: 1.5, bgcolor: 'rgba(255,255,255,0.05)' }}>
                            <Typography variant="caption" color="gray">Duration</Typography>
                            <Typography variant="h6">{Math.round(routeData.estimated_duration_hours)}h</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={6}>
                         <Paper sx={{ p: 1.5, bgcolor: 'rgba(255,255,255,0.05)' }}>
                            <Typography variant="caption" color="gray">Distance</Typography>
                            <Typography variant="h6">{Math.round(routeData.total_distance_km)}km</Typography>
                        </Paper>
                    </Grid>
                </Grid>

                <Alert 
                    severity={routeData.ice_risk_score > 50 ? "warning" : "success"}
                    sx={{ bgcolor: 'transparent', border: '1px solid', borderColor: routeData.ice_risk_score > 50 ? '#ff9800' : '#4caf50' }}
                >
                    Risk Score: {routeData.ice_risk_score}/100
                    <br/>
                    {routeData.ice_risk_score > 50 ? "Significant ice traversal detected." : "Waterway largely clear."}
                </Alert>
            </Box>
        )}
      </Paper>
    </Box>
  );
}
