from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import traceback
from datetime import datetime
from typing import List, Dict, Optional

from app.models import (
    CropInput, CropRecommendationResponse, CropScore, 
    NPKStatus, SoilAnalysis
)
from app.config import get_settings
from app.utils.weather import fetch_current_weather, determine_season_from_month

BASE_DIR = Path(__file__).resolve().parent.parent
KARNATAKA_CROPS_DB_PATH = BASE_DIR / "data" / "karnataka-district-crops.json"

# Load settings
settings = get_settings()

# Initialize FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="AI-powered crop recommendation system for Karnataka farmers - District-specific"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if not settings.CORS_ALLOW_ALL else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Karnataka district crops database at startup
print("📦 Loading Karnataka district-wise crops database...")
try:
    with open(KARNATAKA_CROPS_DB_PATH, "r", encoding="utf-8") as f:
        karnataka_db = json.load(f)
    districts = {d["id"]: d for d in karnataka_db.get("districts", [])}
    print(f"✅ Loaded {len(districts)} Karnataka districts with crops successfully")
except Exception as e:
    print(f"❌ Failed to load Karnataka crops database: {e}")
    districts = {}

# Normalize and augment crops: ensure climate_requirements.seasons exists and normalize region names
for d in list(districts.values()):
    # normalize district id and name to lowercase keys already used
    loc = d.get("location", {})
    if "region" in loc and isinstance(loc["region"], str):
        loc["region"] = loc["region"].strip()

    for crop in d.get("major_crops", []):
        clim = crop.setdefault("climate_requirements", {})
        # if explicit seasons list missing, populate from crop-level season
        if not clim.get("seasons"):
            cs = crop.get("season")
            if cs:
                clim["seasons"] = [cs]
        # normalize seasons to lowercase strings
        if isinstance(clim.get("seasons"), list):
            clim["seasons"] = [s.lower() for s in clim.get("seasons") if isinstance(s, str)]


def get_district_from_region(region: str) -> Optional[Dict]:
    """Get district data by searching district name or region"""
    if not region:
        return None
    region_lower = region.lower().strip()

    for district_id, district_data in districts.items():
        # match id or name
        name = district_data.get("name", "").lower()
        kannada = district_data.get("kannada_name", "").lower()
        did = str(district_id).lower()
        loc_region = str(district_data.get("location", {}).get("region", "")).lower()

        if (
            region_lower == name
            or region_lower == kannada
            or region_lower == did
            or region_lower == loc_region
            or region_lower in name
            or region_lower in loc_region
            or region_lower in did
        ):
            return district_data
    
    return None


def get_npk_status(value: float, nutrient_type: str) -> str:
    """Categorize NPK levels"""
    if nutrient_type == "N":
        if value < 40:
            return "Low"
        elif value < 80:
            return "Medium"
        else:
            return "High"
    elif nutrient_type == "P":
        if value < 20:
            return "Low"
        elif value < 60:
            return "Medium"
        else:
            return "High"
    elif nutrient_type == "K":
        if value < 40:
            return "Low"
        elif value < 80:
            return "Medium"
        else:
            return "High"
    return "Medium"


def get_ph_status(value: float) -> str:
    """Categorize pH levels"""
    if value < 6.0:
        return "Acidic"
    elif value > 7.5:
        return "Alkaline"
    else:
        return "Optimal (Neutral)"


def score_range(value: float, min_val: float, max_val: float) -> float:
    """Normalize value within range to 0-1 score"""
    if value < min_val:
        deficit = min_val - value
        return max(0, 1 - (deficit / min_val * 0.5))
    if value > max_val:
        excess = value - max_val
        return max(0, 1 - (excess / max_val * 0.3))
    return 1.0


def score_crop_for_soil_climate(crop: Dict, n: float, p: float, k: float, ph: float, 
                                soil_type: str, moisture: str, temp: float, 
                                rainfall: float, humidity: float, season: str) -> float:
    """
    Score a crop based on soil and climate match
    
    Weights:
    - NPK matching: 25%
    - pH: 10%
    - Soil type: 15%
    - Moisture: 10%
    - Temperature: 20%
    - Rainfall: 10%
    - Humidity: 5%
    - Season: 5%
    """
    
    soil_req = crop.get("soil_requirements", {})
    clim_req = crop.get("climate_requirements", {})
    
    # NPK Scores
    n_score = score_range(n, soil_req.get("min_nitrogen", 0), soil_req.get("max_nitrogen", 200))
    p_score = score_range(p, soil_req.get("min_phosphorus", 0), soil_req.get("max_phosphorus", 100))
    k_score = score_range(k, soil_req.get("min_potassium", 0), soil_req.get("max_potassium", 200))
    npk_score = (n_score * 0.4 + p_score * 0.3 + k_score * 0.3)
    
    # pH Score
    ph_score = score_range(ph, soil_req.get("min_ph", 0), soil_req.get("max_ph", 14))
    
    # Soil Type Score
    allowed_soils = [s.lower() for s in soil_req.get("soil_types", [])]
    soil_type_lower = soil_type.lower()
    if soil_type_lower in allowed_soils:
        soil_score = 1.0
    elif any(soil_type_lower in s or s in soil_type_lower for s in allowed_soils):
        soil_score = 0.6
    else:
        soil_score = 0.3
    
    # Moisture Score
    allowed_moisture = [m.lower() for m in soil_req.get("moisture_types", [])]
    moisture_score = 1.0 if moisture.lower() in allowed_moisture else 0.4
    
    # Climate Scores
    temp_score = score_range(temp, clim_req.get("min_temperature", 0), clim_req.get("max_temperature", 50))
    
    rainfall_min = clim_req.get("min_rainfall", 0)
    if rainfall < rainfall_min:
        rain_score = max(0, rainfall / rainfall_min * 0.5) if rainfall_min > 0 else 0.5
    elif rainfall > rainfall_min * 3:
        rain_score = 0.7
    else:
        rain_score = 1.0
    
    humidity_score = score_range(humidity, clim_req.get("min_humidity", 0), clim_req.get("max_humidity", 100))
    
    # Use season list from climate_requirements when available.
    # If not provided, fall back to the crop's top-level `season` field.
    season_allowed = [s.lower() for s in clim_req.get("seasons", [])]
    if not season_allowed:
        crop_season = crop.get("season")
        if crop_season:
            season_allowed = [crop_season.lower()]
    season_score = 1.0 if (season and season.lower() in season_allowed) else 0.2

    # Weighted final score (season weight increased to make season more influential)
    final_score = (
        npk_score * 0.22 +
        ph_score * 0.08 +
        soil_score * 0.12 +
        moisture_score * 0.09 +
        temp_score * 0.20 +
        rain_score * 0.08 +
        humidity_score * 0.06 +
        season_score * 0.15
    )
    
    return final_score


def recommend_crops_for_district(
    district_name: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph_level: float,
    soil_type: str,
    soil_moisture: str,
    temperature: float,
    rainfall: float,
    humidity: float,
    season: str,
) -> List[Dict]:
    """
    Recommend crops based on district-specific data + soil/climate conditions
    """
    
    # Get district data
    district = get_district_from_region(district_name)
    if not district:
        return []
    
    recommendations = []
    
    # Score each crop in the district
    for crop in district.get("major_crops", []):
        score = score_crop_for_soil_climate(
            crop, nitrogen, phosphorus, potassium, ph_level,
            soil_type, soil_moisture, temperature, rainfall, humidity, season
        )
        
        if score >= 0.35:  # Include reasonable matches
            recommendations.append({
                "rank": crop.get("rank", 999),
                "name": crop.get("name", ""),
                "kannada_name": crop.get("kannada_name", ""),
                "season": crop.get("season", ""),
                "area_ha": crop.get("area_ha", 0),
                "production_tonnes": crop.get("production_tonnes", 0),
                "yield_kg_ha": crop.get("yield_kg_ha", 0),
                "score": round(score, 3),
            })
    
    # Sort by score descending (then by production for ties)
    recommendations.sort(key=lambda x: (-x["score"], -x["production_tonnes"]))
    return recommendations[:10]  # Top 10


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "RaithaMarga API (Karnataka District-Specific) is running",
        "districts_loaded": len(districts)
    }


@app.post("/api/crop/recommend", response_model=CropRecommendationResponse)
def recommend_crop(payload: CropInput):
    """
    Karnataka District-Specific Crop Recommendation
    
    Recommends crops based on:
    1. District (fetched from region/location)
    2. NPK levels and soil properties
    3. Climate (temperature, rainfall, humidity, season)
    
    Uses actual district production data (2020-21) to prioritize top crops
    """
    
    if not districts:
        raise HTTPException(
            status_code=500,
            detail="Karnataka crops database not loaded"
        )
    
    try:
        # Auto-fetch weather if frontend provided location or climate fields are missing
        temp = payload.temperature
        humidity = payload.humidity
        rainfall = payload.rainfall
        season = payload.season
        weather_source = None

        if (payload.latitude is not None and payload.longitude is not None) and (
            temp is None or humidity is None or rainfall is None or season is None
        ):
            try:
                w = fetch_current_weather(payload.latitude, payload.longitude, target_hour=getattr(payload, 'forecast_hour', None))
                if w:
                    weather_source = "live"
                if temp is None:
                    temp = w.get("temperature")
                if humidity is None:
                    humidity = w.get("humidity")
                if rainfall is None:
                    rainfall = w.get("rainfall")
                if season is None:
                    month = datetime.utcnow().month
                    season = determine_season_from_month(month)
                # If region not provided or generic, prefer city from weather as region
                if (not payload.region or payload.region.strip().lower() == "karnataka") and w.get("city"):
                    payload.region = w.get("city")
            except Exception as e:
                print(f"Warning: failed to fetch weather for provided coords: {e}")
                weather_source = None

        # If frontend didn't provide lat/lon but we know the district/region,
        # attempt to fetch live weather using the district's stored coordinates.
        if (payload.latitude is None or payload.longitude is None) and (
            temp is None or humidity is None or rainfall is None or season is None
        ):
            try:
                district_for_weather = get_district_from_region(payload.region)
                if district_for_weather:
                    loc = district_for_weather.get("location", {})
                    dlat = loc.get("latitude")
                    dlon = loc.get("longitude")
                    if dlat is not None and dlon is not None:
                        try:
                            w2 = fetch_current_weather(float(dlat), float(dlon), target_hour=getattr(payload, 'forecast_hour', None))
                            if w2:
                                weather_source = "live-district-coords"
                            if temp is None:
                                temp = w2.get("temperature")
                            if humidity is None:
                                humidity = w2.get("humidity")
                            if rainfall is None:
                                rainfall = w2.get("rainfall")
                            if season is None:
                                season = determine_season_from_month(datetime.utcnow().month)
                            if (not payload.region or payload.region.strip().lower() == "karnataka") and w2.get("city"):
                                payload.region = w2.get("city")
                        except Exception as e:
                            print(f"Warning: failed to fetch weather from district coords: {e}")
            except Exception:
                pass

        # Fallback to district climate averages if still missing
        district = get_district_from_region(payload.region)
        if district:
            district_climate = district.get("climate", {})
            if temp is None:
                temp = district_climate.get("avg_temperature")
            if rainfall is None:
                rainfall = district_climate.get("avg_rainfall")
            if humidity is None:
                humidity = district_climate.get("avg_humidity", 50)
            if season is None:
                season = district_climate.get("season") or determine_season_from_month(datetime.utcnow().month)
            if weather_source is None:
                weather_source = "district_avg"

        # Ensure we have numeric defaults if still None
        temp = float(temp) if temp is not None else 25.0
        rainfall = float(rainfall) if rainfall is not None else 0.0
        humidity = float(humidity) if humidity is not None else 50.0
        season = season or determine_season_from_month(datetime.utcnow().month)

        # Debug log: show which weather source was used
        try:
            print(f"Weather source used: {weather_source}; temp={temp}, humidity={humidity}, rainfall={rainfall}, region={payload.region}")
        except Exception:
            pass

        # Get recommendations specific to the district (prefer explicit `district`)
        district_name_for_recommend = getattr(payload, 'district', None) or payload.region
        recommendations = recommend_crops_for_district(
            district_name=district_name_for_recommend,
            nitrogen=payload.nitrogen,
            phosphorus=payload.phosphorus,
            potassium=payload.potassium,
            ph_level=payload.ph_level,
            soil_type=payload.soil_type,
            soil_moisture=payload.soil_moisture,
            temperature=temp,
            rainfall=rainfall,
            humidity=humidity,
            season=season,
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=400,
                detail=f"No suitable crops found for {payload.region} with given soil/climate conditions. Try a different district or adjust soil properties."
            )
        
        # Build top 5
        top_5 = [
            CropScore(crop=r["name"], probability=r["score"])
            for r in recommendations[:5]
        ]
        
        # Soil analysis
        soil_analysis = SoilAnalysis(
            nitrogen=NPKStatus(
                value=payload.nitrogen,
                status=get_npk_status(payload.nitrogen, "N")
            ),
            phosphorus=NPKStatus(
                value=payload.phosphorus,
                status=get_npk_status(payload.phosphorus, "P")
            ),
            potassium=NPKStatus(
                value=payload.potassium,
                status=get_npk_status(payload.potassium, "K")
            ),
            ph_level=NPKStatus(
                value=payload.ph_level,
                status=get_ph_status(payload.ph_level)
            ),
        )
        
        return CropRecommendationResponse(
            best_crop=recommendations[0]["name"],
            confidence_score=float(recommendations[0]["score"]),
            top_5_crops=top_5,
            soil_analysis=soil_analysis,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in recommendation: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


# Frontend-compatible aliases (many variants used by the React app during development)
from fastapi import Body


async def _parse_request_to_cropinput(request: Request):
    """Helper: read JSON or form payload and return dict suitable for CropInput"""
    content_type = (request.headers.get("content-type") or "").lower()
    body = {}
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}
    else:
        # Try form data (including when frontend sends application/x-www-form-urlencoded or FormData)
        try:
            form = await request.form()
            body = dict(form)
        except Exception:
            # fallback: try json
            try:
                body = await request.json()
            except Exception:
                body = {}
    # Normalize common frontend wrappers: if payload nested under a single key, unwrap it
    if isinstance(body, dict) and "nitrogen" not in body:
        # keys that may wrap the real payload
        for k in ("values", "payload", "data", "body", "form"):
            if k in body and isinstance(body[k], (dict,)):
                body = body[k]
                break
        else:
            # if body has exactly one dict-valued entry, use it
            dict_vals = [v for v in body.values() if isinstance(v, dict)]
            if len(dict_vals) == 1:
                body = dict_vals[0]
    # Coerce empty strings to None and convert numeric strings to floats where appropriate
    numeric_keys = [
        "nitrogen", "phosphorus", "potassium", "ph_level",
        "temperature", "humidity", "rainfall", "latitude", "longitude"
    ]
    for k in numeric_keys:
        if k in body:
            v = body.get(k)
            if v == "" or v is None:
                body[k] = None
                continue
            # If it's already a number, keep it
            if isinstance(v, (int, float)):
                continue
            # Try converting strings like '50' or '50.0'
            if isinstance(v, str):
                s = v.strip()
                if s == "":
                    body[k] = None
                    continue
                try:
                    # allow integers and floats
                    if "." in s:
                        body[k] = float(s)
                    else:
                        body[k] = float(s)
                except Exception:
                    # leave as-is; CropInput validation will catch invalid types
                    body[k] = v
    return body


@app.post("/crop/recommend", response_model=CropRecommendationResponse)
async def crop_recommend_alias(request: Request):
    body = await _parse_request_to_cropinput(request)
    try:
        ci = CropInput(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return recommend_crop(ci)


@app.post("/crops/recommend", response_model=CropRecommendationResponse)
async def crops_recommend_alias(request: Request):
    body = await _parse_request_to_cropinput(request)
    try:
        ci = CropInput(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return recommend_crop(ci)


@app.post("/crop/recommendations", response_model=CropRecommendationResponse)
async def crop_recommendations_alias(request: Request):
    body = await _parse_request_to_cropinput(request)
    try:
        ci = CropInput(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return recommend_crop(ci)


@app.post("/crops/recommendations", response_model=CropRecommendationResponse)
async def crops_recommendations_alias(request: Request):
    body = await _parse_request_to_cropinput(request)
    try:
        ci = CropInput(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return recommend_crop(ci)


@app.get("/api/districts")
def get_all_districts():
    """Get list of all Karnataka districts"""
    return {
        "total": len(districts),
        "districts": [
            {
                "id": d["id"],
                "name": d["name"],
                "kannada_name": d.get("kannada_name", ""),
                "region": d.get("location", {}).get("region", ""),
                "crops_count": len(d.get("major_crops", []))
            }
            for d in districts.values()
        ]
    }


@app.get("/api/district/{district_id}")
def get_district_crops(district_id: str):
    """Get all major crops for a specific district"""
    district = districts.get(district_id.lower())
    if not district:
        raise HTTPException(
            status_code=404,
            detail=f"District '{district_id}' not found"
        )
    
    return {
        "district": district["name"],
        "kannada_name": district.get("kannada_name", ""),
        "region": district.get("location", {}).get("region", ""),
        "climate": district.get("climate", {}),
        "major_crops": [
            {
                "rank": c.get("rank"),
                "name": c.get("name"),
                "kannada_name": c.get("kannada_name", ""),
                "season": c.get("season", ""),
                "area_ha": c.get("area_ha"),
                "production_tonnes": c.get("production_tonnes"),
                "yield_kg_ha": c.get("yield_kg_ha")
            }
            for c in district.get("major_crops", [])
        ]
    }


@app.get("/api/weather")
def get_weather(lat: float = None, lon: float = None, region: str = None, hour: Optional[int] = None):
    """
    Fetch live weather and determine season
    
    Call Open-Meteo API for current temperature, rainfall, humidity
    Determine season from current month
    """
    # If lat/lon provided, always attempt a live fetch and return immediately.
    try:
        if lat is not None and lon is not None:
            try:
                w = fetch_current_weather(lat, lon, target_hour=hour)
                temperature = w.get("temperature")
                humidity = w.get("humidity")
                rainfall = w.get("rainfall")
                city = w.get("city")
                source = "live"
            except Exception as e:
                print(f"Weather fetch error in endpoint: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch live weather: {e}")

            # Determine season using server month (agricultural season mapping)
            month = datetime.now().month
            if 6 <= month <= 10:
                season = "Kharif"
            elif month in [4, 5]:
                season = "Zaid"
            else:
                season = "Rabi"

            return {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "season": season,
                "lat": lat,
                "lon": lon,
                "target_hour": hour,
                "source": source,
                "city": city,
                "timestamp": datetime.now().isoformat()
            }

        # If lat/lon not provided but region supplied, attempt to resolve to district coords.
        if region:
            district_for_weather = get_district_from_region(region)
            if district_for_weather:
                loc = district_for_weather.get("location", {})
                dlat = loc.get("latitude")
                dlon = loc.get("longitude")
                if dlat is not None and dlon is not None:
                    try:
                        w = fetch_current_weather(float(dlat), float(dlon), target_hour=hour)
                        temperature = w.get("temperature")
                        humidity = w.get("humidity")
                        rainfall = w.get("rainfall")
                        city = w.get("city")
                        source = "live-district-coords"
                    except Exception as e:
                        print(f"Weather fetch error for region->district coords: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to fetch live weather for region: {e}")

                    month = datetime.now().month
                    if 6 <= month <= 10:
                        season = "Kharif"
                    elif month in [4, 5]:
                        season = "Zaid"
                    else:
                        season = "Rabi"

                    return {
                        "temperature": temperature,
                        "humidity": humidity,
                        "rainfall": rainfall,
                        "season": season,
                        "lat": dlat,
                        "lon": dlon,
                        "target_hour": hour,
                        "source": source,
                        "city": city,
                        "timestamp": datetime.now().isoformat()
                    }

        # Neither coordinates nor valid region supplied.
        raise HTTPException(status_code=400, detail="Provide either lat/lon coordinates or a valid region name")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Weather fetch error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather: {str(e)}")



@app.post("/api/weather/coords")
async def weather_coords(request: Request):
    """Return live weather immediately when frontend posts `{"lat":..., "lon":..., "hour":...}`.
    Useful so frontend can get weather as soon as the user allows geolocation.
    """
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            try:
                form = await request.form()
                body = dict(form)
            except Exception:
                body = {}

        lat = body.get("lat") or body.get("latitude")
        lon = body.get("lon") or body.get("longitude")
        hour = body.get("hour") or body.get("forecast_hour")

        if lat is None or lon is None:
            raise HTTPException(status_code=422, detail="lat and lon are required in JSON body")

        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid lat/lon values")

        w = fetch_current_weather(lat, lon, target_hour=int(hour) if hour is not None else None)
        month = datetime.now().month
        if 6 <= month <= 10:
            season = "Kharif"
        elif month in [4, 5]:
            season = "Zaid"
        else:
            season = "Rabi"

        return {
            "temperature": w.get("temperature"),
            "humidity": w.get("humidity"),
            "rainfall": w.get("rainfall"),
            "season": season,
            "lat": lat,
            "lon": lon,
            "target_hour": hour,
            "source": "live",
            "city": w.get("city"),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /api/weather/coords: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/location/find-nearest")
async def find_nearest_district(latitude: float = Query(None), longitude: float = Query(None), request: Request = None):
    """Find nearest district based on GPS coordinates (Haversine distance).

    Accepts either query params (`?latitude=...&longitude=...`) or JSON body {"latitude":.., "longitude":..}.
    Returns the district record and distance in kilometers.
    """
    try:
        # If not provided via query params, try to read JSON body
        if latitude is None or longitude is None:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, dict):
                if latitude is None:
                    latitude = body.get("latitude")
                if longitude is None:
                    longitude = body.get("longitude")

        if latitude is None or longitude is None:
            raise HTTPException(status_code=422, detail="latitude and longitude are required")

        # Haversine formula
        from math import radians, sin, cos, asin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
            dlat = rlat2 - rlat1
            dlon = rlon2 - rlon1
            a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            return 6371.0 * c  # Earth radius in km

        nearest = None
        nearest_dist = None

        for d in districts.values():
            loc = d.get("location", {})
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            if lat is None or lon is None:
                continue
            dist = haversine(float(latitude), float(longitude), float(lat), float(lon))
            if nearest is None or dist < nearest_dist:
                nearest = d
                nearest_dist = dist

        if not nearest:
            raise HTTPException(status_code=404, detail="Could not find nearest district")

        return {
            "id": nearest.get("id"),
            "name": nearest.get("name"),
            "kannada_name": nearest.get("kannada_name"),
            "region": nearest.get("location", {}).get("region"),
            "distance_km": round(nearest_dist, 3),
        }

    except HTTPException:
        raise
    except Exception:
        print(f"Error finding nearest district: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error finding nearest district")


@app.get("/districts")
def districts_alias():
    """Alias for /api/districts (frontend compatibility)."""
    return get_all_districts()


@app.get("/districts/{district_id}")
def district_by_id_alias(district_id: str):
    """Alias for frontend path /districts/{district_id} -> uses existing get_district_crops"""
    return get_district_crops(district_id)


@app.get("/districts/{district_id}/details")
def district_details_alias(district_id: str):
    """Alias for frontend path /districts/{district_id}/details"""
    return get_district_crops(district_id)


@app.get("/districts/details/{district_id}")
def district_details_alias2(district_id: str):
    """Alias for frontend path /districts/details/{district_id}"""
    return get_district_crops(district_id)


@app.get("/regions")
def get_regions():
    """Return list of unique regions extracted from district data."""
    regions = sorted({d.get("location", {}).get("region", "") for d in districts.values() if d.get("location", {}).get("region")})
    return {"total": len(regions), "regions": [{"name": r} for r in regions]}


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "RaithaMarga Crop Recommendation API",
        "version": "2.0.0-Karnataka",
        "description": "District-specific crop recommendations for Karnataka farmers",
        "features": [
            "District-wise crop recommendations",
            "Real-time weather integration",
            "Soil and climate analysis",
            "Kannada language support",
            "Automatic season detection"
        ],
        "endpoints": {
            "health": "/health",
            "recommend": "POST /api/crop/recommend",
            "districts": "GET /api/districts",
            "district_crops": "GET /api/district/{district_id}",
            "weather": "GET /api/weather?lat=LAT&lon=LON",
            "docs": "/docs"
        }
    }
