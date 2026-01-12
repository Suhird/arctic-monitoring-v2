import React, { useState, useRef } from 'react';
import Map, { Source, Layer, NavigationControl, ScaleControl } from 'react-map-gl';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { Box, Paper, Typography, TextField, IconButton, Select, MenuItem, FormControl, Collapse } from '@mui/material';
import { DateRange, ChevronLeft, ChevronRight, ExpandMore, ExpandLess } from '@mui/icons-material';

// @ts-ignore
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || '';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function SatelliteExplorerMap() {
  const [viewState, setViewState] = useState({
    longitude: -40,
    latitude: 75, // Focus on Greenland/Arctic
    zoom: 3
  });

  const [selectedDate, setSelectedDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [source, setSource] = useState('sentinel');
  const [isExpanded, setIsExpanded] = useState(true);

  const handleDateChange = (days: number) => {
    const date = new Date(selectedDate);
    date.setDate(date.getDate() + days);
    const newDate = date.toISOString().split('T')[0];
    console.log("Date changed to:", newDate);
    setSelectedDate(newDate);
  };
  
  // Debug Log
  console.log("Current Source URL:", `${API_URL}/api/v1/satellite/tiles/{z}/{x}/{y}?date=${selectedDate}&source=${source}`);

  return (
    <Box sx={{ width: '100%', height: '85vh', position: 'relative', bgcolor: '#000' }}>
      
      {/* Floating Controls - Repositioned to top-right */}
      <Paper 
        elevation={4}
        sx={{ 
           position: 'absolute', 
           top: 20, 
           right: 20, 
           zIndex: 1000, 
           p: 2, 
           bgcolor: 'rgba(20, 25, 40, 0.95)', 
           color: 'white',
           backdropFilter: 'blur(10px)',
           border: '1px solid rgba(255,255,255,0.1)',
           display: 'flex',
           flexDirection: 'column',
           gap: 2,
           minWidth: isExpanded ? 280 : 'auto',
           maxWidth: 320,
           transition: 'all 0.3s ease'
        }}
      >
         <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
               <DateRange sx={{ color: '#4facfe' }} />
               <Typography variant="h6" sx={{ fontWeight: 'bold' }}>Satellite</Typography>
            </Box>
            <IconButton 
               onClick={() => setIsExpanded(!isExpanded)} 
               size="small" 
               sx={{ color: 'white' }}
            >
               {isExpanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
         </Box>

         <Collapse in={isExpanded}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
               {/* Source Selector */}
               <FormControl variant="standard" fullWidth>
                  <Select
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    sx={{ 
                        color: 'white', 
                        fontSize: '0.9rem',
                        '&:before': { borderColor: 'rgba(255,255,255,0.3)' }, 
                        '&:after': { borderColor: '#4facfe' },
                        '& .MuiSvgIcon-root': { color: 'white' } 
                    }}
                  >
                     <MenuItem value="sentinel">Sentinel-1 (Radar)</MenuItem>
                     <MenuItem value="maxar">Maxar (Premium Optical)</MenuItem>
                  </Select>
               </FormControl>

               {/* Date Control */}
               <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', bgcolor: 'rgba(255,255,255,0.05)', p: 1, borderRadius: 1 }}>
                   <IconButton onClick={() => handleDateChange(-1)} size="small" sx={{ color: 'white' }}>
                      <ChevronLeft />
                   </IconButton>
                   
                   <TextField
                      type="date"
                      variant="standard"
                      value={selectedDate}
                      onChange={(e) => setSelectedDate(e.target.value)}
                      sx={{ 
                          input: { color: 'white', textAlign: 'center' },
                          width: 140
                      }}
                      InputProps={{
                          disableUnderline: true
                      }}
                   />
                   
                   <IconButton onClick={() => handleDateChange(1)} size="small" sx={{ color: 'white' }}>
                      <ChevronRight />
                   </IconButton>
               </Box>

               <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                   <Box sx={{ width: 12, height: 12, bgcolor: '#ffffff', border: '1px solid #aaa' }} />
                   <Typography variant="caption">Ice / Land</Typography>
               </Box>
               <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                   <Box sx={{ width: 12, height: 12, bgcolor: '#004', border: '1px solid #aaa' }} />
                   <Typography variant="caption">Water</Typography>
               </Box>
            </Box>
         </Collapse>

      </Paper>

      <Map
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        style={{ width: '100%', height: '100%' }}
        // Use a satellite or dark style
        mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        mapboxAccessToken={process.env.REACT_APP_MAPBOX_TOKEN}
        projection={{ name: 'globe' }} // Enable Globe view like Copernicus
        minZoom={4} // Restrict zoom to match Sentinel-1 resolution limits
      >
        <NavigationControl position="top-right" />
        <ScaleControl />

        {/* Real-Time Satellite Raster Layer */}
        {/* We use key={selectedDate} to force re-mounting source when date changes, 
            ensuring tiles refresh immediately without caching stale date */}
        <Source 
            key={`${selectedDate}-${source}`}
            id="sentinel-source" 
            type="raster" 
            tiles={[`${API_URL}/api/v1/satellite/tiles/{z}/{x}/{y}?date=${selectedDate}&source=${source}`]} 
            tileSize={256}
        >
            <Layer 
            id="sentinel-layer" 
            type="raster" 
            paint={{ 
                "raster-opacity": 1.0, 
                "raster-fade-duration": 300 
            }} 
            />
        </Source>

        {/* Ice Concentration Layer: Visualizes sea ice concentration (0-100%) */}
        {/* Style: White-to-Blue Gradient (NSIDC style) */}
        <Source
            id="ice-concentration-source"
            type="geojson"
            data={`${API_URL}/api/v1/ice/current?min_lon=-180&min_lat=50&max_lon=180&max_lat=90&date=${selectedDate}`}
        >
            <Layer
                id="ice-concentration-layer"
                type="fill"
                paint={{
                    'fill-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'concentration'],
                        0, 'rgba(0,0,255,0)',       // 0%: Transparent
                        15, 'rgba(0,100,255,0.4)',  // 15%: Blue (Ice Edge)
                        50, 'rgba(100,200,255,0.6)',// 50%: Light Blue
                        90, 'rgba(230,240,255,0.9)',// 90%: Very Light Blue
                        100, '#ffffff'              // 100%: White
                    ],
                    'fill-opacity': 0.8
                }}
                beforeId="land-mask-layer" // Ensure it renders UNDER land mask (so land covers ice if overlap)
            />
        </Source>
        
        {/* Land Mask Layer: Overlays land areas to hide radar data over land */}
        {/* We use 'mapbox.country-boundaries-v1' which provides polygons for all countries/land masses */}
        <Source id="land-mask-source" type="vector" url="mapbox://mapbox.country-boundaries-v1">
            <Layer
                id="land-mask-layer"
                type="fill"
                source-layer="country_boundaries"
                paint={{
                    'fill-color': '#0B1026', // Deep dark blue/black to match ocean/space theme
                    'fill-opacity': 1.0
                }}
            />
        </Source>

        {/* Optional: Fog for realistic atmosphere */}
        <Layer 
            id="sky-atmosphere"
            type="sky"
            paint={{
                'sky-type': 'atmosphere',
                'sky-atmosphere-sun': [0.0, 0.0],
                'sky-atmosphere-sun-intensity': 15
            }}
        />

      </Map>
    </Box>
  );
}


