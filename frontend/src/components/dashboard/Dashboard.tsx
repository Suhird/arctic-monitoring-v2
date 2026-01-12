/**
 * Main Dashboard Component
 */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Button,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Logout as LogoutIcon,
} from '@mui/icons-material';
import { useAuthStore } from '../../store/authStore';
import IceConcentrationMap from '../map/IceConcentrationMap';
import PredictionsMap from '../map/PredictionsMap';
import VesselRoutingMap from '../map/VesselRoutingMap';
import PermafrostMap from '../map/PermafrostMap';
import MapboxDebug from '../MapboxDebug';

import SatelliteImagery from './SatelliteImagery';

export default function Dashboard() {
  const [selectedView, setSelectedView] = useState('map');
  const user = useAuthStore((state) => state.user);
  const logout = useAuthStore((state) => state.logout);
  const navigate = useNavigate();

  const handleTabChange = (_: React.SyntheticEvent, newValue: string) => {
    if (newValue === 'routing') {
        navigate('/routing');
    } else {
        setSelectedView(newValue);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* App Bar with Tabs */}
      <AppBar position="static" sx={{ zIndex: 1201 }}>
        <Toolbar>
          <Typography variant="h6" sx={{ mr: 4, fontWeight: 'bold' }}>
            Arctic Monitor
          </Typography>
          
          <Box sx={{ flexGrow: 1 }}>
            <Tabs 
                value={selectedView === 'routing' ? false : selectedView} 
                onChange={handleTabChange}
                textColor="inherit"
                indicatorColor="secondary"
            >
                <Tab label="Ice Concentration Map" value="map" />
                <Tab label="Satellite Imagery" value="satellite" />
                <Tab label="7-Day Predictions" value="predictions" />
                <Tab label="Vessel Routing" value="routing" />
                <Tab label="Permafrost Analysis" value="permafrost" />
            </Tabs>
          </Box>

          <Typography variant="body2" sx={{ mr: 2 }}>
            {user?.email}
          </Typography>
          <Button color="inherit" startIcon={<LogoutIcon />} onClick={logout}>
            Logout
          </Button>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1, overflow: 'hidden', position: 'relative' }}>
         {/* Render Active View */}
        {selectedView === 'map' && <IceConcentrationMap />}
        {selectedView === 'satellite' && <SatelliteImagery />}
        {selectedView === 'predictions' && <PredictionsMap />}
        {selectedView === 'routing' && <VesselRoutingMap />}
        {selectedView === 'permafrost' && <PermafrostMap />}
        <MapboxDebug />
      </Box>
    </Box>
  );
}
