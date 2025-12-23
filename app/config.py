from functools import lru_cache
import os


class Settings:
    API_TITLE = "RaithaMarga Crop Recommendation API"
    API_VERSION = "1.0.0"
    
    # CORS Origins - Frontend domains allowed to access API
    CORS_ORIGINS = [
        "http://localhost:3000",      # React dev server
        "http://localhost:3001",      # Alternative React port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]
    
    # Allow all origins (for Render/production - use with caution)
    CORS_ALLOW_ALL = os.getenv("CORS_ALLOW_ALL", "false").lower() == "true"
    
    # Weather API Configuration
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
    WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "open-meteo")  # open-meteo is free!
    
    # Database/Optional settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./crops.db")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (FastAPI best practice)"""
    return Settings()
