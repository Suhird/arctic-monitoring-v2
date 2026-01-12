"""
Arctic Ice Monitoring Platform - FastAPI Application
"""
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import requests
import os
from pathlib import Path
from .config import settings
from .database import init_db
from .api import auth, ice_concentration, predictions, routing, permafrost, historical, satellite, ice_stats, changes, ice_classification

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time Arctic ice monitoring with ML-powered predictions",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(ice_concentration.router)
app.include_router(ice_stats.router)
app.include_router(predictions.router)
app.include_router(routing.router)
app.include_router(permafrost.router)
app.include_router(historical.router)
app.include_router(satellite.router, prefix="/api/v1/satellite", tags=["satellite"])
app.include_router(changes.router)
app.include_router(ice_classification.router)

# Cache Directory
CACHE_DIR = Path("data/cache/bremen")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")






@app.get("/")
def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs"
    }


@app.get("/api/v1/proxy/bremen_visual")
async def proxy_bremen_visual(date: str, style: str = "visual"):
    """
    Proxy Bremen visual images to bypass CORS, with caching.
    Format: YYYY-MM-DD
    """
    try:
        # Check cache
        cache_file = CACHE_DIR / f"{date}_{style}.png"
        if cache_file.exists():
             return FileResponse(cache_file, media_type="image/png")

        dt = datetime.strptime(date, "%Y-%m-%d")
        yyyy = dt.strftime("%Y")
        mm = dt.strftime("%b").lower()
        yyyymmdd = dt.strftime("%Y%m%d")
        
        # Construct upstream URL
        url = f"https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250/{yyyy}/{mm}/Arctic/asi-AMSR2-n6250-{yyyymmdd}-v5.4_{style}.png"
        
        print(f"Proxying (Miss): {url}")
        
        # Fetch with a standard User-Agent to avoid blocking
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        
        if response.status_code == 200:
            print(f"SUCCESS: Fetched image. Saving to {cache_file}")
            with open(cache_file, "wb") as f:
                f.write(response.content)
            return FileResponse(cache_file, media_type="image/png")
        else:
            print(f"Upstream {response.status_code} for {url}. Returning transparent placeholder.")
            # Return 1x1 transparent PNG
            transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            return Response(content=transparent_png, media_type="image/png")
            
    except Exception as e:
        print(f"Proxy Error: {e}")
        # Return transparent PNG on error too
        transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        return Response(content=transparent_png, media_type="image/png")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
