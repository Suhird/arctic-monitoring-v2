
import React from 'react';
import { Box } from '@mui/material';
import SatelliteExplorerMap from '../map/SatelliteExplorerMap';

export default function SatelliteImagery() {
  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <SatelliteExplorerMap />
    </Box>
  );
}
