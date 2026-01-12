/**
 * Type definitions for Arctic Ice Platform
 */

export interface User {
  id: string;
  email: string;
  full_name?: string;
  role: 'free' | 'paid' | 'admin';
  api_key?: string;
  created_at: string;
}

export interface AuthTokens {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface IceFeature {
  type: 'Feature';
  geometry: {
    type: 'Polygon';
    coordinates: number[][][];
  };
  properties: {
    concentration_percent: number;
    ice_type: string;
    satellite_source: string;
    timestamp: string;
  };
}

export interface IceConcentrationData {
  type: 'FeatureCollection';
  features: IceFeature[];
}

export interface PredictionDay {
  date: string;
  predicted_concentration: number;
  confidence: number;
  spatial_data: number[][];
}

export interface VesselRoute {
  route_id: number;
  path: {
    type: 'LineString';
    coordinates: number[][];
  };
  waypoints: Array<{ lat: number; lon: number }>;
  ice_risk_score: number;
  estimated_duration_hours: number;
  total_distance_km: number;
}

export interface PermafrostSite {
  site_id: number;
  site_name: string;
  site_type: string;
  stability_score: number;
  temperature_c: number;
  alert_level: 'stable' | 'warning' | 'critical';
  last_analysis: string;
  recommendations?: string[];
}

export interface BoundingBox {
  min_lon: number;
  min_lat: number;
  max_lon: number;
  max_lat: number;
}

export interface IceExtent {
  date: string;
  extent_km2: number;
  area_km2: number;
}

export interface IceThicknessFeature {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
  properties: {
    thickness_m: number;
    uncertainty: number;
    source: string;
    timestamp: string;
  };
}

export interface IceThicknessData {
  type: "FeatureCollection";
  features: IceThicknessFeature[];
}

export interface IceMotionFeature {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
  properties: {
    u_velocity: number;
    v_velocity: number;
    velocity_mag: number;
    source: string;
    timestamp: string;
  };
}

export interface IceMotionData {
  type: "FeatureCollection";
  features: IceMotionFeature[];
}

export interface RouteRequest {
  start: { lat: number; lon: number };
  end: { lat: number; lon: number };
  vesselType: string;
  departureTime: string;
}


export interface RouteResponse {
  route_id: number;
  path: {
    type: "LineString";
    coordinates: number[][];
  };
  ice_risk_score: number;
  estimated_duration_hours: number;
  total_distance_km: number;
}
