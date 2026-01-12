/**
 * Ice Concentration Map Component
 * Visualizing AMSR2 Sea Ice Concentration from University of Bremen
 */
import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, CircularProgress, Button, TextField, AppBar, Toolbar, Tabs, Tab, IconButton } from '@mui/material';
import { ChevronLeft, ChevronRight } from '@mui/icons-material';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

// @ts-ignore
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || '';

// Coordinates derived from gdalinfo on the n6250 grid
const BREMEN_COORDINATES = [
  [168.3497, 30.9795], // Top-Left
  [102.3391, 31.3642], // Top-Right
  [-9.9721, 34.3444],  // Bottom-Right
  [-80.7398, 33.9239]  // Bottom-Left
];

export default function IceConcentrationMap() {
  // Default to yesterday as data is usually 1 day behind
  const [selectedDate, setSelectedDate] = useState<string>(() => {
    const d = new Date();
    d.setDate(d.getDate() - 1);
    return d.toISOString().split('T')[0];
  });
  
  // UI State
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentTab, setCurrentTab] = useState(0); // 0 = Explorer, 1 = Split View
  
  // Map State
  const [overlayMode, setOverlayMode] = useState(false); // Default to Vector (Stable)
  const [visualStyle, setVisualStyle] = useState<'visual' | 'nic'>('visual');
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Generate last 7 days
  const last7Days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - i);
    return d.toISOString().split('T')[0];
  });

  const getBremenVisualUrl = (dateStr: string) => {
    return `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/v1/proxy/bremen_visual?date=${dateStr}&style=${visualStyle}`;
  };

  useEffect(() => {
    // Re-initialize map if container exists and map doesn't (or if view changed)
    if (!mapContainer.current || map.current) return;
    
    console.log("Initializing Mapbox...");
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/outdoors-v12', 
      center: [-40, 85], 
      zoom: 2.5, 
      maxZoom: 6, 
      projection: { name: 'globe' } as any,
    });

    map.current.on('load', () => {
      setMapLoaded(true);
      map.current?.setFog({
          'range': [0.5, 10], 'color': 'white', 'horizon-blend': 0.3,
          'high-color': '#add8e6', 'space-color': '#d8f2ff', 'star-intensity': 0.0
      });
    });

    // Cleanup on unmount is tricky in React 18 with refs, 
    // strictly we should only remove if component unmounts, not re-renders.
    // For now, allow map to persist.

  }, []); // Run once on mount. 
  
  // NOTE: If we switch tabs and unmount the map container, we lose the map instance.
  // Strategy: Keep map always mounted but hidden/resized? 
  // OR: Destory and recreate map on tab switch? Recreating is cleaner for layout but slower.
  // Given "Split View" needs the map alongside image, we can just resizing the map container.

  // Handle Layers
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    const sourceId = 'ice-concentration-daily';
    const imageSourceId = 'ice-image-overlay';
    
    const dataUrl = `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/v1/ice/current?min_lon=-180&min_lat=50&max_lon=180&max_lat=90&date=${selectedDate}`;
    const imageUrl = getBremenVisualUrl(selectedDate);

    // 1. Vector Source
    if (map.current.getSource(sourceId)) {
        (map.current.getSource(sourceId) as mapboxgl.GeoJSONSource).setData(dataUrl);
    } else {
        map.current.addSource(sourceId, { type: 'geojson', data: dataUrl });
    }

    // 2. Image Source
    if (map.current.getSource(imageSourceId)) {
        (map.current.getSource(imageSourceId) as mapboxgl.ImageSource).updateImage({ url: imageUrl, coordinates: BREMEN_COORDINATES });
    } else {
        map.current.addSource(imageSourceId, {
            type: 'image',
            url: imageUrl,
            coordinates: BREMEN_COORDINATES
        });
    }

    // Layer Management
    const heatmapId = 'ice-heatmap';
    const rasterId = 'ice-raster-layer';

    if (overlayMode) {
        if (map.current.getLayer(heatmapId)) map.current.setLayoutProperty(heatmapId, 'visibility', 'none');
        
        if (!map.current.getLayer(rasterId)) {
            map.current.addLayer({
                id: rasterId,
                type: 'raster',
                source: imageSourceId,
                paint: { 'raster-opacity': 0.8, 'raster-fade-duration': 0 }
            });
        } else {
            map.current.setLayoutProperty(rasterId, 'visibility', 'visible');
        }
    } else {
        if (map.current.getLayer(rasterId)) map.current.setLayoutProperty(rasterId, 'visibility', 'none');
        
        if (!map.current.getLayer(heatmapId)) {
            map.current.addLayer({
                id: heatmapId,
                type: 'heatmap',
                source: sourceId,
                maxzoom: 9, 
                paint: {
                    'heatmap-weight': ['interpolate', ['linear'], ['get', 'concentration'], 0, 0, 100, 1],
                    'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
                    'heatmap-color': [
                        'interpolate', ['linear'], ['heatmap-density'],
                        0, 'rgba(0,0,255,0)',
                        0.2, 'rgba(0,0,255, 0.5)',
                        0.5, 'rgba(0,255,255, 0.6)',
                        1.0, 'rgba(255,255,255, 0.9)'
                    ],
                    'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 2, 9, 20],
                    'heatmap-opacity': 0.8
                }
            });
        } else {
             map.current.setLayoutProperty(heatmapId, 'visibility', 'visible');
        }
    }
    
    // Resize map when sidebar/tabs change to ensure it fills container
    map.current.resize();

  }, [selectedDate, mapLoaded, overlayMode, visualStyle, sidebarOpen, currentTab]);

  // Auto-align Map in Split View
  useEffect(() => {
    if (currentTab === 1 && map.current) {
        // Align to Arctic view matching Bremen image
        // The Bremen image covers the entire Arctic cap down to ~30N.
        // We fit the bounds of the "extent".
        // Using a fixed bearing of 0 to match the static image orientation.
        map.current.flyTo({
            center: [0, 90],
            zoom: 1.5, // Zoom out to see the whole cap
            pitch: 0,
            bearing: 0,
            essential: true
        });
    }
  }, [currentTab]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#f5f5f5', overflow: 'hidden' }}>
      
      {/* 1. Sub-Header & Tabs */}
      <Paper elevation={0} sx={{ zIndex: 1200, borderBottom: '1px solid #e0e0e0' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', px: 2 }}>
            <Tabs 
                value={currentTab} 
                onChange={(_: React.SyntheticEvent, v: number) => setCurrentTab(v)} 
                indicatorColor="primary" 
                textColor="primary"
                sx={{ minHeight: 48 }}
            >
                <Tab label="Explorer Map" sx={{ minHeight: 48 }} />
                <Tab label="Split Comparison" sx={{ minHeight: 48 }} />
            </Tabs>
        </Box>
      </Paper>

      {/* 2. Main Layout (Sidebar + Content) */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        
        {/* Collapsible Sidebar */}
        <Paper 
            elevation={2}
            sx={{ 
                width: sidebarOpen ? 260 : 0, 
                transition: 'width 0.3s',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
                display: 'flex',
                flexDirection: 'column',
                borderRight: '1px solid #ddd',
                zIndex: 1000
            }}
        >
             <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: '#f0f0f0' }}>
                <Typography variant="subtitle2" fontWeight="bold" sx={{ color: 'black' }}>CONTROLS</Typography>
                <IconButton size="small" onClick={() => setSidebarOpen(false)}><ChevronLeft /></IconButton>
             </Box>

             <Box sx={{ p: 2, overflowY: 'auto', flex: 1 }}>
                 {/* Date Picker */}
                 <Typography variant="caption" fontWeight="bold" gutterBottom sx={{ color: 'black' }}>DATE SELECTION</Typography>
                 <Box sx={{ display: 'grid', gap: 1, mb: 3 }}>
                    {last7Days.map(date => (
                        <Button
                            key={date}
                            variant={selectedDate === date ? "contained" : "outlined"}
                            size="small"
                            onClick={() => setSelectedDate(date)}
                            sx={{ justifyContent: 'flex-start' }}
                        >
                            {date}
                        </Button>
                    ))}
                 </Box>

                 {/* Image Style - Always visible for Split View Reference */}
                 <Typography variant="caption" fontWeight="bold" gutterBottom sx={{ display: 'block', mb: 1, mt: 2, color: 'black' }}>IMAGE STYLE (Ref / Overlay)</Typography>
                 <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                    <Button 
                        size="small" 
                        variant={visualStyle === 'visual' ? "contained" : "outlined"} 
                        onClick={() => setVisualStyle('visual')} 
                        color="secondary"
                        sx={{ flex: 1 }}
                    >
                        Visual (RGB)
                    </Button>
                    <Button 
                        size="small" 
                        variant={visualStyle === 'nic' ? "contained" : "outlined"} 
                        onClick={() => setVisualStyle('nic')} 
                        color="secondary"
                        sx={{ flex: 1 }}
                    >
                        NIC (Chart)
                    </Button>
                 </Box>

                 {/* Layer Settings */}
                 <Typography variant="caption" fontWeight="bold" gutterBottom sx={{ display: 'block', mb: 1, pl: 1, color: '#000000' }}>MAP LAYERS</Typography>
                 <Button 
                    variant={overlayMode ? "contained" : "outlined"} 
                    color="primary"
                    fullWidth
                    size="small"
                    onClick={() => setOverlayMode(!overlayMode)}
                    sx={{ mb: 1, textAlign: 'left', justifyContent: 'flex-start' }}
                >
                    {overlayMode ? "Image Overlay (Active)" : "Vector Heatmap (Active)"}
                </Button>
             </Box>
        </Paper>

        {/* Sidebar Toggle Button (When Closed) - High Contrast */}
        {!sidebarOpen && (
             <Box sx={{ position: 'absolute', top: 20, left: 0, zIndex: 1100 }}>
                <IconButton 
                    onClick={() => setSidebarOpen(true)}
                    sx={{ 
                        bgcolor: 'primary.main', 
                        color: 'white',
                        borderRadius: '0 4px 4px 0', 
                        boxShadow: 2,
                        '&:hover': { bgcolor: 'primary.dark' } 
                    }}
                >
                    <ChevronRight />
                </IconButton>
             </Box>
        )}

        {/* Content Area */}
        <Box sx={{ flex: 1, display: 'flex', position: 'relative' }}>
            
            {/* Map Pane (Always rendered to preserve WebGL context, just resized/hidden) */}
            <Box sx={{ 
                flex: 1, 
                position: 'relative', 
                borderRight: currentTab === 1 ? '1px solid #ccc' : 'none'
            }}>
                 <Box ref={mapContainer} sx={{ width: '100%', height: '100%' }} />
            </Box>

            {/* Split View Right Pane (Only if Tab 1) */}
            {currentTab === 1 && (
                <Box sx={{ 
                    flex: 1, 
                    bgcolor: '#222', // Slightly lighter than pure black for contrast 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    position: 'relative', 
                    overflow: 'hidden',
                    minWidth: 0,  // Prevent flex item from overflowing
                    minHeight: 0,
                    p: 4, // Increased padding
                    boxSizing: 'border-box'
                }}>
                    <img 
                        src={getBremenVisualUrl(selectedDate)} 
                        alt="Bremen Visual Reference" 
                        style={{ maxWidth: '100%', maxHeight: '90%', objectFit: 'contain' }}
                    />
                     <Box sx={{ position: 'absolute', top: 10, left: 10, bgcolor: 'rgba(255,255,255,0.9)', p: 0.5, px: 1, borderRadius: 1 }}>
                        <Typography variant="caption" fontWeight="bold" color="black">Reference: {visualStyle.toUpperCase()}</Typography>
                    </Box>
                </Box>
            )}

        </Box>

      </Box>
    </Box>
  );
}
