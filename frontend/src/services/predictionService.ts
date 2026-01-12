/**
 * Prediction and routing service
 */
import api from './api';
import { VesselRoute, PermafrostSite } from '../types';

export const predictionService = {
  async getVesselRoute(
    startCoords: { lat: number; lon: number },
    endCoords: { lat: number; lon: number },
    vesselType: string = 'cargo'
  ): Promise<VesselRoute> {
    const response = await api.post('/api/v1/routing/recommend', {
      start_coords: startCoords,
      end_coords: endCoords,
      vessel_type: vesselType,
    });
    return response.data;
  },

  async analyzePermafrost(
    location: { lat: number; lon: number },
    siteType: string,
    siteName: string
  ): Promise<PermafrostSite> {
    const response = await api.post('/api/v1/permafrost/analyze', {
      location,
      site_type: siteType,
      site_name: siteName,
    });
    return response.data;
  },

  async getPermafrostSites(): Promise<PermafrostSite[]> {
    const response = await api.get('/api/v1/permafrost/sites');
    return response.data.sites;
  },

  async getPermafrostAlerts(): Promise<PermafrostSite[]> {
    const response = await api.get('/api/v1/permafrost/alerts');
    return response.data.alerts;
  },
};
