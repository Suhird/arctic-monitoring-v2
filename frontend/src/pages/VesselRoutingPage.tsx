/**
 * Vessel Routing Page
 * "Google Maps for the Arctic" style interface
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  TextField, 
  Button, 
  MenuItem, 
  CircularProgress,
  Divider,
  Grid,
  Chip,
  IconButton,
  Card,
  CardContent,
  Alert
} from '@mui/material';
import { 
  Navigation, 
  AccessTime, 
  Explore, 
  Warning, 
  DirectionsBoat,
  Menu as MenuIcon,
  ArrowBack
} from '@mui/icons-material';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { useNavigate } from 'react-router-dom';
import { iceService } from '../services/iceService';
import { RouteResponse } from '../types';
import { generateIceContours } from '../utils/contourUtils';

// @ts-ignore
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || '';

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

export default function VesselRoutingPage() {
  const navigate = useNavigate();
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Form State
  const [startPort, setStartPort] = useState('Nome, AK');
  const [endPort, setEndPort] = useState('Barrow, AK');
  const [vesselType, setVesselType] = useState('icebreaker_light');
  
  // Route State
  const [iceData, setIceData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [routeData, setRouteData] = useState<RouteResponse | null>(null);
  const [error, setError] = useState('');

  const [useRealData, setUseRealData] = useState(true);

  // Fetch Real-time Ice Data
  useEffect(() => {
    const fetchIceData = async () => {
        try {
            // Fetch current ice concentration (full arctic)
            const data = await iceService.getCurrentIceData({
                min_lon: -180, min_lat: 60, max_lon: 180, max_lat: 90
            }, useRealData); 
            setIceData(data);
        } catch (e) {
            console.error("Failed to load initial ice context", e);
        }
    };
    fetchIceData();
  }, [useRealData]);

  // Map Initialization
  useEffect(() => {
    if (!mapContainer.current || map.current) return;
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-100, 75],
      zoom: 3,
      projection: { name: 'globe' } as any,
    });

    map.current.on('load', () => {
      setMapLoaded(true);
      // Optional: Add ice layer background here if desired
    });

    return () => map.current?.remove();
  }, []);

  // Render Ice Layer (Contours)
  useEffect(() => {
    if (mapLoaded && iceData && map.current) {
        
        // Generate Contours
        const contourData = generateIceContours(iceData);

        if (!map.current.getSource('ice-contours') && contourData) {
            map.current.addSource('ice-contours', {
                type: 'geojson',
                data: contourData as any
            });

            try {
                const beforeId = map.current.getLayer('route-line') ? 'route-line' : undefined;
                
                // Add Fill Layer for Contours
                map.current.addLayer({
                    id: 'ice-contour-fill',
                    type: 'fill',
                    source: 'ice-contours',
                    paint: {
                        'fill-color': ['get', 'fillColor'],
                        'fill-opacity': 0.6,
                        'fill-outline-color': 'rgba(255,255,255,0.1)' // faint outline for distinct bands
                    }
                }, beforeId);
                console.log("Ice contour layer added successfully");
            } catch (err) {
                console.error("Error adding contour layer:", err);
            }
        }
    }
  }, [mapLoaded, iceData]);

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
      
      // Visualize Route
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

    // Cleanup previous
    if (map.current.getSource('route')) {
      map.current.removeLayer('route-line');
      map.current.removeLayer('route-points');
      map.current.removeSource('route');
    }

    // Add Source
    map.current.addSource('route', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: [
           {
            type: 'Feature',
            properties: {},
            geometry: route.path
           }
        ]
      }
    });

    // Add Line Layer
    map.current.addLayer({
      id: 'route-line',
      type: 'line',
      source: 'route',
      layout: { 'line-join': 'round', 'line-cap': 'round' },
      paint: {
        'line-color': [
          'case',
          ['>', route.ice_risk_score, 70], '#ff1744', // Red for high risk
          ['>', route.ice_risk_score, 30], '#ff9100', // Orange for medium
          '#00e676' // Green for safe
        ],
        'line-width': 5
      }
    });

    // Fit Bounds
    const coords = route.path.coordinates;
    const bounds = new mapboxgl.LngLatBounds(coords[0] as any, coords[0] as any);
    coords.forEach(c => bounds.extend(c as [number, number]));
    map.current.fitBounds(bounds, { padding: 100 });
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh', bgcolor: '#0a1929', color: 'white' }}>
      
      {/* SIDEBAR */}
      <Paper sx={{ 
        width: 400, 
        zIndex: 10, 
        display: 'flex', 
        flexDirection: 'column',
        borderRight: '1px solid rgba(255,255,255,0.1)',
        bgcolor: '#0a1929'
      }}>
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2, borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <IconButton onClick={() => navigate('/dashboard')} sx={{ color: 'white' }}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h6" fontWeight="bold">
            Vessel Routing
          </Typography>
        </Box>

        <Box sx={{ p: 3, flex: 1, overflowY: 'auto' }}>
          
          <Typography variant="subtitle2" color="gray" gutterBottom>
            VOYAGE PARAMETERS
          </Typography>

          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12}>
              <TextField
                select
                fullWidth
                label="Departure Port"
                value={startPort}
                onChange={(e) => setStartPort(e.target.value)}
                size="small"
                sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}
              >
                {Object.keys(PORTS).map(k => <MenuItem key={k} value={k}>{k}</MenuItem>)}
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                select
                fullWidth
                label="Destination Port"
                value={endPort}
                onChange={(e) => setEndPort(e.target.value)}
                size="small"
                sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}
              >
                {Object.keys(PORTS).map(k => <MenuItem key={k} value={k}>{k}</MenuItem>)}
              </TextField>
            </Grid>
            <Grid item xs={12}>
               <TextField
                select
                fullWidth
                label="Vessel Class"
                value={vesselType}
                onChange={(e) => setVesselType(e.target.value)}
                size="small"
                helperText="Determines speed and ice capability"
                sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}
              >
                <MenuItem value="icebreaker_heavy">Polar Class 1 (Heavy Icebreaker)</MenuItem>
                <MenuItem value="icebreaker_light">Polar Class 4 (Light Icebreaker)</MenuItem>
                <MenuItem value="merchant_reinforced">Ice Class 1A (Reinforced Merchant)</MenuItem>
              </TextField>
            </Grid>
          </Grid>

          <Button
            fullWidth
            variant={useRealData ? "contained" : "outlined"}
            size="small"
            onClick={() => setUseRealData(!useRealData)}
            sx={{ mb: 3, borderColor: 'rgba(255,255,255,0.3)', color: 'white', bgcolor: useRealData ? 'primary.main' : 'transparent' }}
          >
            {useRealData ? "Source: Real Satellite (NSIDC)" : "Source: Simulation (Demo)"}
          </Button>

          <Button 
            fullWidth 
            variant="contained" 
            size="large"
            onClick={calculateRoute}
            disabled={!mapLoaded || loading}
            startIcon={loading ? <CircularProgress size={20} color="inherit"/> : <Navigation />}
            sx={{ mb: 3, py: 1.5, fontSize: '1.1rem' }}
          >
            {loading ? "Calculating..." : "Find Optimal Route"}
          </Button>

          {error && (
            <Paper sx={{ p: 2, bgcolor: 'rgba(255, 23, 68, 0.1)', border: '1px solid #ff1744', mb: 2 }}>
              <Typography color="#ff1744" variant="body2">{error}</Typography>
            </Paper>
          )}

          {routeData && (
            <Box>
              <Typography variant="subtitle2" color="gray" gutterBottom>
                ROUTE ANALYSIS
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <Card sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}>
                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                      <Typography variant="caption" color="gray">Est. Duration</Typography>
                      <Typography variant="h5" color="white" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <AccessTime fontSize="small" />
                        {Math.round(routeData.estimated_duration_hours)}h
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}>
                     <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                      <Typography variant="caption" color="gray">Distance</Typography>
                      <Typography variant="h5" color="white" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Explore fontSize="small" />
                        {Math.round(routeData.total_distance_km)}km
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              <Paper sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.05)', mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="gray">Ice Risk Score</Typography>
                  <Chip 
                    label={`${routeData.ice_risk_score}/100`} 
                    size="small" 
                    color={routeData.ice_risk_score > 50 ? "error" : "success"} 
                  />
                </Box>
                <Typography variant="caption" color="gray">
                  Based on current concentration, thickness, and vessel capability.
                </Typography>
              </Paper>

              <Alert 
                severity={routeData.ice_risk_score > 0 ? "warning" : "info"}
                sx={{ bgcolor: 'transparent', color: 'white' }}
                icon={<Warning sx={{ color: routeData.ice_risk_score > 0 ? '#ff9800' : '#2196f3' }} />}
              >
                {routeData.ice_risk_score > 50 
                  ? "Route traverses defined ice zones. Heavy thickness detected. Proceed with extreme caution."
                  : "Route is relatively clear of heavy ice. Standard monitoring advised."}
              </Alert>
            </Box>
          )}

        </Box>
      </Paper>

      {/* MAP AREA */}
      <Box sx={{ flex: 1, position: 'relative' }}>
         {!mapLoaded && (
          <Box sx={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: '#0a1929' }}>
            <CircularProgress />
            <Typography sx={{ ml: 2 }}>Initializing Satellite Data...</Typography>
          </Box>
         )}
         <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />
      </Box>

    </Box>
  );
}
