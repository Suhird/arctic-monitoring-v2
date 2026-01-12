/**
 * Ice data service
 */
import api from './api';
import { IceConcentrationData, BoundingBox } from '../types';

export const iceService = {
  async getCurrentIceData(bbox: BoundingBox, useRealData: boolean = false, date?: string, source?: string): Promise<IceConcentrationData> {
    const endpoint = useRealData ? '/api/v1/ice/realtime' : '/api/v1/ice/current';
    const response = await api.get(endpoint, {
      params: {
        min_lon: bbox.min_lon,
        min_lat: bbox.min_lat,
        max_lon: bbox.max_lon,
        max_lat: bbox.max_lat,
        date: date,
        source: source
      },
    });
    return response.data;
  },

  async getTimeSeries(bbox: BoundingBox, startDate: Date, endDate: Date) {
    const response = await api.get('/api/v1/ice/timeseries', {
      params: {
        min_lon: bbox.min_lon,
        min_lat: bbox.min_lat,
        max_lon: bbox.max_lon,
        max_lat: bbox.max_lat,
        start_date: startDate.toISOString(),
        end_date: endDate.toISOString(),
      },
    });
    return response.data;
  },

  async get7DayPrediction(bbox: BoundingBox) {
    const response = await api.get('/api/v1/predictions/7day', {
      params: {
        min_lon: bbox.min_lon,
        min_lat: bbox.min_lat,
        max_lon: bbox.max_lon,
        max_lat: bbox.max_lat,
      },
    });
    return response.data;
  },

  async getHistoricalPatterns(bbox: BoundingBox, startYear: number, endYear: number) {
    const response = await api.get('/api/v1/historical/patterns', {
      params: {
        min_lon: bbox.min_lon,
        min_lat: bbox.min_lat,
        max_lon: bbox.max_lon,
        max_lat: bbox.max_lat,
        start_year: startYear,
        end_year: endYear,
      },
    });
    return response.data;
  },
  async getIceExtentHistory(days: number = 30): Promise<any[]> {
    const response = await api.get('/api/v1/stats/extent', {
      params: { days }
    });
    return response.data;
  },

  async getIceThickness(): Promise<any> {
    const response = await api.get('/api/v1/stats/thickness');
    return response.data;
  },

  async getIceMotion(): Promise<any> {
    const response = await api.get('/api/v1/stats/motion');
    return response.data;
  },

  async calculateRoute(start: {lat: number, lon: number}, end: {lat: number, lon: number}, vesselType: string) {
    const response = await api.post('/api/v1/routing/calculate', {
      start_coords: start,
      end_coords: end,
      vessel_type: vesselType,
      departure_time: new Date().toISOString()
    });
    return response.data;
  },

  async getSatelliteSources() {
    const response = await api.get('/api/v1/satellite/sources');
    return response.data;
  },

  async detectChanges(image1: File, image2: File) {
    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);
    const response = await api.post('/api/v1/ice/changes', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async classifyImage(image: File) {
    const formData = new FormData();
    formData.append('file', image);
    const response = await api.post('/api/v1/ice/classify', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  getSatelliteImageUrl(productId: string): string {
    // Return direct URL to endpoint (browser will fetch it)
    return `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/v1/satellite/image/${productId}`;
  },

  async listSatelliteProducts(bbox: BoundingBox, startDate?: string, endDate?: string) {
    const response = await api.get('/api/v1/satellite/products', {
      params: {
        min_lon: bbox.min_lon,
        min_lat: bbox.min_lat,
        max_lon: bbox.max_lon,
        max_lat: bbox.max_lat,
        start_date: startDate,
        end_date: endDate
      }
    });
    return response.data;
  }
};
