/**
 * 7-Day Ice Predictions Map Component
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, Paper, Typography, CircularProgress, Alert, Slider, Button } from '@mui/material';
import { PlayArrow, Pause } from '@mui/icons-material';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { iceService } from '../../services/iceService';
import { BoundingBox } from '../../types';

// @ts-ignore
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || '';

const DEFAULT_BBOX: BoundingBox = {
  min_lon: -180,
  min_lat: 60,
  max_lon: 180,
  max_lat: 90,
};

export default function PredictionsMap() {
  const [predictionsData, setPredictionsData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedDay, setSelectedDay] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const addPredictionsToMap = useCallback((data: any, day: number) => {
    if (!map.current || !data) return;

    // Remove existing layers and sources
    if (map.current.getSource('predictions-data')) {
      if (map.current.getLayer('predictions-fill')) {
        map.current.removeLayer('predictions-fill');
      }
      if (map.current.getLayer('predictions-outline')) {
        map.current.removeLayer('predictions-outline');
      }
      map.current.removeSource('predictions-data');
    }

    // Filter features for selected day
    const dayFeatures = data.features.filter((f: any) => f.properties.forecast_days === day);
    const dayData = { ...data, features: dayFeatures };

    // Add predictions source
    map.current.addSource('predictions-data', {
      type: 'geojson',
      data: dayData as any,
    });

    // Add fill layer
    map.current.addLayer({
      id: 'predictions-fill',
      type: 'fill',
      source: 'predictions-data',
      paint: {
        'fill-color': [
          'interpolate',
          ['linear'],
          ['get', 'predicted_concentration'],
          0, '#0000ff',
          25, '#4169e1',
          50, '#87ceeb',
          75, '#e0ffff',
          100, '#ffffff'
        ],
        'fill-opacity': [
          'interpolate',
          ['linear'],
          ['get', 'confidence_score'],
          0, 0.3,
          0.5, 0.5,
          1.0, 0.8
        ],
      },
    });

    // Add outline
    map.current.addLayer({
      id: 'predictions-outline',
      type: 'line',
      source: 'predictions-data',
      paint: {
        'line-color': '#ffff00',
        'line-width': 2,
        'line-opacity': 0.8,
      },
    });

    // Add popup
    map.current.on('click', 'predictions-fill', (e) => {
      if (!e.features || e.features.length === 0) return;
      const props = e.features[0].properties;
      new mapboxgl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(`
          <strong>Day ${props?.forecast_days} Prediction</strong><br/>
          <strong>Predicted Concentration:</strong> ${props?.predicted_concentration?.toFixed(1)}%<br/>
          <strong>Confidence:</strong> ${(props?.confidence_score * 100)?.toFixed(1)}%<br/>
          <strong>Model:</strong> ${props?.model_version || 'N/A'}
        `)
        .addTo(map.current!);
    });

    // Change cursor
    map.current.on('mouseenter', 'predictions-fill', () => {
      if (map.current) map.current.getCanvas().style.cursor = 'pointer';
    });
    map.current.on('mouseleave', 'predictions-fill', () => {
      if (map.current) map.current.getCanvas().style.cursor = '';
    });
  }, []);

  // Fetch current real-time ice data for background context
  const [iceContextData, setIceContextData] = useState<any>(null);

  const loadPredictions = useCallback(async () => {
    setLoading(true);
    try {
      const [predData, contextData] = await Promise.all([
        iceService.get7DayPrediction(DEFAULT_BBOX),
        iceService.getCurrentIceData(DEFAULT_BBOX, true) // Fetch real satellite data
      ]);
      setPredictionsData(predData);
      setIceContextData(contextData);
      setError('');
    } catch (err: any) {
      setError('Failed to load prediction data. Please check if the backend is running.');
      console.error('Error loading predictions:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPredictions();
  }, [loadPredictions]);

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-135, 75],
      zoom: 3,
      projection: { name: 'globe' } as any,
    });

    const currentMap = map.current;

    return () => {
      currentMap.remove();
    };
  }, []); // Only initialize once

  useEffect(() => {
    if (!predictionsData || !map.current) return;

    const currentMap = map.current;

    if (currentMap.loaded()) {
      if (iceContextData) addIceContextLayer(iceContextData);
      addPredictionsToMap(predictionsData, selectedDay);
    } else {
      currentMap.on('load', () => {
        if (iceContextData) addIceContextLayer(iceContextData);
        addPredictionsToMap(predictionsData, selectedDay);
      });
    }
  }, [predictionsData, iceContextData, selectedDay, addPredictionsToMap]);

  const addIceContextLayer = (data: any) => {
    if (!map.current || !data) return;
    
    if (!map.current.getSource('ice-context')) {
        map.current.addSource('ice-context', {
            type: 'geojson',
            data: data
        });
        
        map.current.addLayer({
            id: 'ice-context-fill',
            type: 'fill',
            source: 'ice-context',
            paint: {
                'fill-color': '#ffffff',
                'fill-opacity': 0.15
            }
        }, 'predictions-fill'); // Render below predictions
    }
  };

  const togglePlay = () => {
    if (isPlaying) {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      playIntervalRef.current = setInterval(() => {
        setSelectedDay((prev) => (prev >= 7 ? 1 : prev + 1));
      }, 1500);
    }
  };

  useEffect(() => {
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, []);

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
          {/* Timeline Control */}
          <Paper sx={{ position: 'absolute', bottom: 32, left: '50%', transform: 'translateX(-50%)', p: 3, minWidth: 400, bgcolor: 'rgba(19, 47, 76, 0.95)', zIndex: 999 }}>
            <Typography variant="h6" gutterBottom>
              Day {selectedDay} Forecast
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Button variant="contained" size="small" onClick={togglePlay} startIcon={isPlaying ? <Pause /> : <PlayArrow />}>
                {isPlaying ? 'Pause' : 'Play'}
              </Button>
              <Slider
                value={selectedDay}
                onChange={(_, value) => setSelectedDay(value as number)}
                min={1}
                max={7}
                step={1}
                marks
                valueLabelDisplay="auto"
                sx={{ flexGrow: 1 }}
              />
            </Box>
          </Paper>

          {/* Info Panel */}
          <Paper sx={{ position: 'absolute', top: 16, right: 16, p: 2, minWidth: 280, maxWidth: 350, bgcolor: 'rgba(19, 47, 76, 0.95)', zIndex: 999 }}>
            <Typography variant="h6" gutterBottom color="primary">
              7-Day Ice Predictions
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Forecast Day:</strong> {selectedDay}/7
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Regions:</strong> {predictionsData?.features.filter((f: any) => f.properties.forecast_days === selectedDay).length || 0}
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Model:</strong> LSTM v1.0
            </Typography>
            <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
              <Typography variant="caption" color="text.secondary">
                Yellow borders indicate predicted ice areas. Opacity shows prediction confidence.
              </Typography>
            </Box>
          </Paper>
        </>
      )}
    </Box>
  );
}
