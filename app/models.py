
from pydantic import BaseModel, Field
from typing import List, Optional

class CropInput(BaseModel):
    """Input schema for crop recommendation"""
    nitrogen: float = Field(..., ge=0, le=200, description="Nitrogen level (N)")
    phosphorus: float = Field(..., ge=0, le=200, description="Phosphorus level (P)")
    potassium: float = Field(..., ge=0, le=200, description="Potassium level (K)")
    ph_level: float = Field(..., ge=0, le=14, description="pH level of soil")
    # Climate inputs can be omitted if frontend provides `latitude`/`longitude`.
    # Backend will auto-fetch current climate when location is provided.
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    rainfall: Optional[float] = Field(None, ge=0, description="Rainfall in mm")
    soil_type: str = Field(default="Loamy", description="Type of soil")
    soil_moisture: str = Field(default="Medium", description="Moisture level")
    season: Optional[str] = Field(None, description="Season (auto-detected if omitted)")
    region: str = Field(default="Karnataka", description="Region/State")
    # Optional location supplied by frontend when user allows geolocation
    latitude: Optional[float] = Field(None, description="Latitude for auto climate fetch")
    longitude: Optional[float] = Field(None, description="Longitude for auto climate fetch")
    # Optional district name provided by frontend (preferred over `region` when present)
    district: Optional[str] = Field(None, description="District name (preferred over region)")
    # Optional: target hour (0-23) to sample today's weather. If omitted, use nearest-to-now.
    forecast_hour: Optional[int] = Field(None, ge=0, le=23, description="Target hour (0-23) to sample today's weather")
    # Note: weather is fetched live for current conditions; do not request future days here.

class CropScore(BaseModel):
    """Individual crop recommendation with score"""
    crop: str
    probability: float

class NPKStatus(BaseModel):
    """NPK nutrient status"""
    value: float
    status: str

class SoilAnalysis(BaseModel):
    """Soil analysis summary"""
    nitrogen: NPKStatus
    phosphorus: NPKStatus
    potassium: NPKStatus
    ph_level: NPKStatus

class CropRecommendationResponse(BaseModel):
    """Final recommendation response"""
    best_crop: str
    confidence_score: float
    top_5_crops: List[CropScore]
    soil_analysis: SoilAnalysis
