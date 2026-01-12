
import React from 'react';
import { Dialog, DialogTitle, DialogContent, IconButton, Box, Typography, Button } from '@mui/material';
import { Close, Download } from '@mui/icons-material';
import { iceService } from '../../services/iceService';

interface IceImageViewerProps {
  open: boolean;
  onClose: () => void;
  productId: string;
  productName?: string;
}

const IceImageViewer: React.FC<IceImageViewerProps> = ({ open, onClose, productId, productName }) => {
  const imageUrl = productId ? iceService.getSatelliteImageUrl(productId) : '';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: '#132f4c', color: 'white' }}>
        <Typography variant="h6">Satellite Image Quicklook</Typography>
        <IconButton onClick={onClose} sx={{ color: 'white' }}>
          <Close />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ bgcolor: '#0a1929', p: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
        {productId ? (
          <>
             <Box sx={{ width: '100%', height: 'calc(80vh - 100px)', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', p: 2 }}>
               <img 
                 src={imageUrl} 
                 alt="Satellite Quicklook" 
                 style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                 onError={(e) => {
                   (e.target as HTMLImageElement).src = 'https://via.placeholder.com/800x600?text=Image+Load+Failed';
                 }}
               />
             </Box>
             <Box sx={{ p: 2, width: '100%', display: 'flex', justifyContent: 'space-between', color: 'rgba(255,255,255,0.7)', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                <Typography variant="body2">Product ID: {productId}</Typography>
                <Typography variant="body2">{productName}</Typography>
                <Button startIcon={<Download />} size="small" href={imageUrl} target="_blank" download>
                    Download
                </Button>
             </Box>
          </>
        ) : (
          <Typography color="white">No Product Selected</Typography>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default IceImageViewer;
