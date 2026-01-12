/**
 * Authentication service
 */
import api from './api';
import { User, AuthTokens } from '../types';

export const authService = {
  async login(email: string, password: string): Promise<AuthTokens> {
    const response = await api.post('/api/v1/auth/login', { email, password });
    const tokens = response.data;
    localStorage.setItem('access_token', tokens.access_token);
    return tokens;
  },

  async register(email: string, password: string, full_name?: string): Promise<User> {
    const response = await api.post('/api/v1/auth/register', {
      email,
      password,
      full_name,
    });
    return response.data;
  },

  async getCurrentUser(): Promise<User> {
    const response = await api.get('/api/v1/auth/me');
    return response.data;
  },

  logout() {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
  },

  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  },
};
