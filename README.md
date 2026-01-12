# Arctic Ice Monitoring Platform

A comprehensive platform for real-time Arctic ice monitoring with ML-powered predictions, satellite data integration, and geospatial analysis.

## Features

- **Real-time Ice Concentration Maps**: Live ice concentration data from multiple satellite sources
- **7-Day Ice Movement Predictions**: LSTM-powered forecasting of ice movement
- **Historical Ice Pattern Analysis**: Analyze seasonal trends and long-term patterns
- **Vessel Routing Recommendations**: Optimal path calculation considering ice conditions
- **Permafrost Stability Analysis**: Monitor permafrost stability around buildings and infrastructure
- **RESTful API**: Full API access for integration with external systems

## Technology Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **PostgreSQL + PostGIS** - Geospatial database
- **Redis** - Caching and real-time features
- **PyTorch + TensorFlow** - Machine learning models
- **Celery** - Background task processing

### Frontend
- **React 18 + TypeScript** - Modern UI framework
- **Material-UI** - Component library
- **Mapbox GL JS** - Map rendering
- **Zustand** - State management

### Satellite Data Sources

#### Free
- **Sentinel-1/2 (ESA)**: SAR and optical imagery (10-20m resolution)
  - Register at: https://scihub.copernicus.eu/dhus

#### Paid (Requires Subscription)
- **RADARSAT Constellation**: C-band SAR (3-100m resolution)
  - URL: https://www.eodms-sgdot.nrcan-rncan.gc.ca/
- **Planet Labs**: High-res optical (3-5m resolution)
  - URL: https://api.planet.com/data/v1
- **Maxar/DigitalGlobe**: Very high-res (30cm-1m resolution)
  - URL: https://securewatch.digitalglobe.com/

## Quick Start

### Prerequisites
- **Python 3.11+** (for local development)
- **Docker & Docker Compose** (for full stack deployment)
- **Git**
- **UV** (recommended - fast Python package manager)

### Option 1: Quick Setup with UV (Recommended for Development)

UV is 10-100x faster than pip! Perfect for local development.

**Install UV and setup environment:**

```bash
# macOS/Linux
./setup_uv.sh

# Windows PowerShell
.\setup_uv.ps1
```

The setup script will:
1. Install UV if not present
2. Create a virtual environment
3. Let you choose which components to install (Backend, ML, or Both)
4. Install all dependencies in ~1 minute

**Manual UV setup:**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv
uv venv

# Activate venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows

# Install all dependencies
uv pip install -e ".[all]"

# Or install specific components
uv pip install -e ".[backend]"  # Backend only
uv pip install -e ".[ml,dev]"   # ML + Jupyter
```

See [UV_SETUP.md](UV_SETUP.md) for detailed instructions.

### Option 2: Docker Deployment (Recommended for Production)

### Prerequisites
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd arctic-ice-platform
```

2. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` and configure:
- `DB_PASSWORD`: Database password
- `JWT_SECRET`: Secret key for JWT tokens (use `openssl rand -base64 32`)
- `SENTINEL_USERNAME` & `SENTINEL_PASSWORD`: Sentinel credentials (free)
- Optional: Add paid satellite API keys

3. **Start all services**
```bash
docker-compose up -d
```

This will start:
- PostgreSQL + PostGIS (port 5432)
- Redis (port 6379)
- Backend API (port 8000)
- Celery worker (background tasks)
- Frontend (port 3000)

4. **Initialize database**
```bash
# The database will be automatically initialized on first run
# Check logs: docker-compose logs backend
```

5. **Access the platform**
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/api/docs
- Backend Health: http://localhost:8000/health

### First Steps

1. Register a new account at http://localhost:3000/register
2. Login with your credentials
3. Explore the ice concentration map
4. View 7-day predictions
5. Test vessel routing and permafrost analysis

## API Documentation

### Authentication Endpoints

**Register User**
```bash
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "John Doe"
}
```

**Login**
```bash
POST /api/v1/auth/login
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### Ice Concentration Endpoints

**Get Current Ice Data**
```bash
GET /api/v1/ice/current?min_lon=-180&min_lat=60&max_lon=-120&max_lat=85
```

**Get Time Series**
```bash
GET /api/v1/ice/timeseries?min_lon=-180&min_lat=60&max_lon=-120&max_lat=85&start_date=2024-01-01T00:00:00Z&end_date=2024-12-31T23:59:59Z
```

### Prediction Endpoints

**Get 7-Day Forecast**
```bash
GET /api/v1/predictions/7day?min_lon=-180&min_lat=60&max_lon=-120&max_lat=85
```

### Vessel Routing

**Calculate Optimal Route**
```bash
POST /api/v1/routing/recommend
{
  "start_coords": {"lat": 70.0, "lon": -140.0},
  "end_coords": {"lat": 75.0, "lon": -130.0},
  "vessel_type": "cargo"
}
```

### Permafrost Analysis

**Analyze Site**
```bash
POST /api/v1/permafrost/analyze
{
  "location": {"lat": 70.0, "lon": -120.0},
  "site_type": "building",
  "site_name": "Research Station Alpha"
}
```

**Get Monitored Sites**
```bash
GET /api/v1/permafrost/sites
```

**Get Alerts**
```bash
GET /api/v1/permafrost/alerts
```

## Machine Learning Models

### Ice Classification CNN
- **Model**: ResNet50 (pre-trained on ImageNet, fine-tuned)
- **Input**: Satellite imagery (256x256 RGB/SAR)
- **Output**: Ice type classification + concentration percentage
- **Classes**: Open water, thin ice, thick ice, multi-year ice, pack ice

### Ice Movement Predictor
- **Model**: Stacked ConvLSTM
- **Input**: 30-day historical ice concentration sequences
- **Output**: 7-day future predictions
- **Architecture**: Spatial-temporal feature extraction

### Change Detection
- **Model**: Siamese Network
- **Purpose**: Detect changes between time periods
- **Output**: Change map highlighting ice movement/melt

## Satellite Data Integration

### Setting Up Sentinel-1/2 (FREE)

1. Register at https://scihub.copernicus.eu/dhus
2. Add credentials to `.env`:
```bash
SENTINEL_USERNAME=your_username
SENTINEL_PASSWORD=your_password
```

3. The platform will automatically fetch Arctic imagery (60°N - 90°N) every 6 hours

### Adding Paid Satellite Sources

Once you have API keys from paid providers, add them to `.env`:
```bash
RADARSAT_API_KEY=your_key
PLANET_API_KEY=your_key
MAXAR_API_KEY=your_key
```

## Development

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm start
```

### Database Migrations

```bash
cd backend
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

## Training ML Models

### Ice Classifier

```bash
cd ml-models
python training/train_ice_classifier.py
```

### LSTM Predictor

```bash
python training/train_lstm_predictor.py
```

## Project Structure

```
arctic-ice-platform/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── models/         # Database models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   ├── ml/             # ML models
│   │   ├── satellite/      # Satellite integrations
│   │   └── utils/          # Utilities
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API clients
│   │   ├── store/         # State management
│   │   └── types/         # TypeScript types
│   ├── package.json
│   └── Dockerfile
├── ml-models/             # ML training scripts
│   ├── training/
│   └── models/
├── docker-compose.yml
├── .env.example
└── README.md
```

## Deployment

### Production Deployment

1. Update `.env` with production values
2. Change `DB_PASSWORD` and `JWT_SECRET` to strong values
3. Configure proper CORS origins
4. Use production-grade web server (Nginx)

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Monitoring

- **Backend logs**: `docker-compose logs -f backend`
- **Database logs**: `docker-compose logs -f postgres`
- **Celery worker logs**: `docker-compose logs -f celery_worker`

## Troubleshooting

### Backend won't start
- Check database is running: `docker-compose ps`
- Check logs: `docker-compose logs backend`
- Ensure `.env` file exists and is configured

### Frontend can't connect to backend
- Verify `REACT_APP_API_URL` in `.env`
- Check CORS configuration in backend
- Ensure backend is running on port 8000

### No ice data showing
- Check if database has been initialized
- Verify Sentinel credentials are configured
- Run manual satellite data ingestion

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file

## Support

For issues and questions:
- GitHub Issues: <your-repo-url>/issues
- Documentation: http://localhost:8000/api/docs

## Acknowledgments

- ESA Copernicus for Sentinel data
- NASA for Arctic climate data
- PostGIS for geospatial database support

  Satellite Data Sources

## Free (Implemented)

  ✅ Sentinel-1/2 (ESA) - 10-20m resolution, Arctic coverage
  - Register: https://scihub.copernicus.eu/dhus
  - Add credentials to .env

## Paid (Placeholder - Ready for API Keys)

  🔑 RADARSAT Constellation - 3-100m resolution
  - URL: https://www.eodms-sgdot.nrcan-rncan.gc.ca/

  🔑 Planet Labs - 3-5m resolution, daily revisit
  - URL: https://api.planet.com/data/v1

  🔑 Maxar/DigitalGlobe - 30cm-1m resolution
  - URL: https://securewatch.digitalglobe.com/

## Features Implemented

  ✅ Real-time ice concentration maps (GeoJSON API)
  ✅ 7-day ML-powered ice movement predictions
  ✅ Historical ice pattern analysis
  ✅ Vessel routing recommendations
  ✅ Permafrost stability monitoring
  ✅ RESTful API with OpenAPI documentation
  ✅ JWT authentication + role-based access
  ✅ Redis caching (5min-1hr TTLs)
  ✅ PostgreSQL + PostGIS for spatial queries
  ✅ ResNet50 ice classification CNN
  ✅ LSTM ice movement predictor
  ✅ Sentinel-1/2 satellite integration
  ✅ Docker Compose deployment

##  Next Steps

  1. Start the platform: `docker-compose up -d`
  2. Register for Sentinel: Get free satellite data at https://scihub.copernicus.eu/dhus
  3. Train ML models: Add labeled Arctic imagery to ml-models/data/ and run training scripts
  4. Configure Mapbox: Get free token at https://www.mapbox.com/ for full map visualization
  5. Add paid satellite APIs: When you get API keys, they're already integrated (just add to .env)
