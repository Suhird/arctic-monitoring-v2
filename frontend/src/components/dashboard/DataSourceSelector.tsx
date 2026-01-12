import React, { useEffect, useState } from 'react';
import { FormControl, Select, MenuItem, InputLabel, CircularProgress, Chip, Box, Typography, Alert } from '@mui/material';
import { iceService } from '../../services/iceService';

interface DataSource {
  id: string;
  name: string;
  type: string;
  status: string;
  description: string;
  requires_key: boolean;
}

interface DataSourceSelectorProps {
  selectedSource: string;
  onSourceChange: (sourceId: string) => void;
  disabled?: boolean;
}

export default function DataSourceSelector({ selectedSource, onSourceChange, disabled = false }: DataSourceSelectorProps) {
  const [sources, setSources] = useState<DataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchSources = async () => {
      try {
        const data = await iceService.getSatelliteSources();
        setSources(data);
        // If no source selected, or current one invalid, select first available
        if (!selectedSource && data.length > 0) {
           onSourceChange(data[0].id);
        }
      } catch (err) {
        setError('Failed to load sources');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchSources();
  }, []);

  if (loading) return <CircularProgress size={20} />;
  if (error) return <Alert severity="error" sx={{py:0}}>{error}</Alert>;

  return (
    <FormControl fullWidth size="small" variant="outlined" disabled={disabled}>
      <InputLabel id="source-select-label" sx={{color: 'rgba(255,255,255,0.7)'}}>Satellite Source</InputLabel>
      <Select
        labelId="source-select-label"
        value={selectedSource || (sources.length > 0 ? sources[0].id : '')}
        onChange={(e) => onSourceChange(e.target.value)}
        label="Satellite Source"
        sx={{
          color: 'white',
          '.MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255,255,255,0.3)' },
          '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255,255,255,0.5)' },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: 'primary.main' },
          '.MuiSvgIcon-root': { color: 'white' }
        }}
        MenuProps={{
            PaperProps: {
                sx: { 
                    bgcolor: '#132f4c', 
                    color: 'white',
                    maxWidth: 300
                }
            }
        }}
      >
        {sources.map((source) => (
          <MenuItem value={source.id} key={source.id} sx={{flexDirection:'column', alignItems:'flex-start', py:1}}>
             <Box sx={{display:'flex', alignItems:'center', width:'100%', mb:0.5}}>
                 <Typography variant="body1">{source.name}</Typography>
                 <Box sx={{flexGrow:1}}/>
                 {source.status !== 'active' && source.status !== 'configured' && (
                     <Chip label="Setup Required" size="small" color="error" sx={{height:16, fontSize:10}} />
                 )}
                 {(source.status === 'active' || source.status === 'configured') && (
                     <Chip label="Ready" size="small" color="success" sx={{height:16, fontSize:10}} />
                 )}
             </Box>
             <Typography variant="caption" sx={{color:'rgba(255,255,255,0.5)'}}>{source.description}</Typography>
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}
