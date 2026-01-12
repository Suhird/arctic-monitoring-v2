/**
 * Mapbox Debug Component - Shows if token is loaded
 */
import React, { useEffect, useState } from 'react';
import { Box, Typography, Alert } from '@mui/material';

export default function MapboxDebug() {
  const [tokenStatus, setTokenStatus] = useState<string>('checking...');

  useEffect(() => {
    const token = process.env.REACT_APP_MAPBOX_TOKEN;

    if (!token) {
      setTokenStatus('❌ No token found in environment');
    } else if (token.startsWith('pk.')) {
      setTokenStatus(`✅ Valid public token loaded (${token.substring(0, 20)}...)`);
    } else if (token.startsWith('sk.')) {
      setTokenStatus('❌ WARNING: Using secret token (should be public pk.)');
    } else {
      setTokenStatus(`⚠️ Unknown token format: ${token.substring(0, 10)}...`);
    }
  }, []);

  return (
    <Box sx={{ position: 'fixed', bottom: 16, left: 16, zIndex: 10000, maxWidth: 400 }}>
      <Alert severity={tokenStatus.includes('✅') ? 'success' : 'error'} variant="filled">
        <Typography variant="caption">
          <strong>Mapbox Status:</strong><br />
          {tokenStatus}
        </Typography>
      </Alert>
    </Box>
  );
}
