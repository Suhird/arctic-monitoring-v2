/**
 * Sea Ice Extent Chart
 * Displays historical sea ice extent using Recharts
 */
import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { Box, Typography, Paper, CircularProgress } from '@mui/material';
import { iceService } from '../../services/iceService';
import { IceExtent } from '../../types';

export default function IceExtentChart() {
  const [data, setData] = useState<IceExtent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const history = await iceService.getIceExtentHistory(30);
        // Format data for chart
        const formatted = history.map(item => ({
          ...item,
          extent_m_km2: (item.extent_km2 / 1_000_000).toFixed(2),
          date: new Date(item.date).toLocaleDateString()
        }));
        setData(formatted);
      } catch (err) {
        console.error('Failed to load extent data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><CircularProgress /></Box>;
  }

  return (
    <Paper sx={{ p: 2, height: 350, bgcolor: 'rgba(19, 47, 76, 0.4)' }}>
      <Typography variant="h6" gutterBottom color="primary">
        Sea Ice Extent (Last 30 Days)
      </Typography>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis 
            dataKey="date" 
            stroke="#90caf9" 
            tick={{ fill: '#90caf9' }}
          />
          <YAxis 
            stroke="#90caf9"
            tick={{ fill: '#90caf9' }} 
            label={{ value: 'Million km²', angle: -90, position: 'insideLeft', fill: '#90caf9' }} 
            domain={['auto', 'auto']}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid #333' }}
            itemStyle={{ color: '#fff' }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="extent_m_km2" 
            name="Ice Extent" 
            stroke="#00bcd4" 
            strokeWidth={3} 
            activeDot={{ r: 8 }} 
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
}
