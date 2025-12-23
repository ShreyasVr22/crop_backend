import requests
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from app.config import get_settings

settings = get_settings()


def determine_season_from_month(month: int) -> str:
    """Return agricultural season name for India based on month number."""
    # Kharif: June(6) - September(9)
    if 6 <= month <= 9:
        return "Kharif"
    # Rabi: October(10) - March(3)
    if month >= 10 or month <= 3:
        return "Rabi"
    # Zaid: April(4) - May(5)
    return "Zaid"


def geocode_city(city: str) -> Optional[Tuple[float, float]]:
    """Use OpenWeatherMap geocoding to resolve city -> (lat, lon)."""
    key = settings.WEATHER_API_KEY
    if not key:
        return None
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": key}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def fetch_current_weather(lat: float, lon: float, target_hour: Optional[int] = None) -> Dict[str, Any]:
    """Fetch current weather for given lat/lon.

    Uses OpenWeatherMap when `WEATHER_API_KEY` is configured, otherwise falls back to Open-Meteo.
    Returns dict with keys: `temperature`, `rainfall`, `humidity`, `city`.
    """
    key = settings.WEATHER_API_KEY
    if key:
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            temp_c = data.get("main", {}).get("temp")
            humidity = data.get("main", {}).get("humidity")
            rain = 0.0
            rain_data = data.get("rain", {}) or {}
            if "1h" in rain_data:
                rain = float(rain_data.get("1h", 0.0))
            elif "3h" in rain_data:
                rain = float(rain_data.get("3h", 0.0))
            city = data.get("name")
            return {"temperature": temp_c, "rainfall": rain, "humidity": humidity, "city": city}
        except Exception:
            # fallback to Open-Meteo below
            pass

    try:
        om_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto"
        )
        r = requests.get(om_url, timeout=10)
        r.raise_for_status()
        data = r.json()

        current = data.get("current_weather", {})
        temp_c = current.get("temperature")

        # Extract latest hourly humidity/precipitation (choose closest hour to now)
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        rh = hourly.get("relativehumidity_2m") or []
        precip = hourly.get("precipitation") or []

        humidity = None
        rainfall = 0.0
        if times:
            from datetime import datetime, timezone
            # Parse times into datetimes and try to select the requested target_hour for today
            parsed = []
            for t in times:
                try:
                    ts = t
                    if ts.endswith("Z"):
                        ts = ts.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    parsed.append(dt)
                except Exception:
                    parsed.append(None)

            best_idx = None
            now = None
            # Use timezone-aware 'now' relative to the datetimes if possible
            for dt in parsed:
                if dt is not None:
                    now = datetime.now(dt.tzinfo)
                    break
            if now is None:
                now = datetime.now(timezone.utc)

            # If target_hour provided, try to find an index matching today's date and that hour
            if target_hour is not None:
                for i, dt in enumerate(parsed):
                    if dt is None:
                        continue
                    try:
                        if dt.date() == now.date() and dt.hour == int(target_hour):
                            best_idx = i
                            break
                    except Exception:
                        continue

            # Fallback: choose index closest to now
            if best_idx is None:
                best_diff = None
                for i, dt in enumerate(parsed):
                    if dt is None:
                        continue
                    try:
                        diff = abs((dt - now).total_seconds())
                        if best_diff is None or diff < best_diff:
                            best_diff = diff
                            best_idx = i
                    except Exception:
                        continue

            if best_idx is None:
                best_idx = len(times) - 1

            try:
                if best_idx < len(rh):
                    humidity = float(rh[best_idx]) if rh[best_idx] not in (None, "") else None
            except Exception:
                humidity = None
            try:
                if best_idx < len(precip):
                    rainfall = float(precip[best_idx]) if precip[best_idx] not in (None, "") else 0.0
            except Exception:
                rainfall = 0.0

            # If humidity missing, try nearby hourly indices (±1, ±2)
            if humidity is None and times:
                for offset in (1, -1, 2, -2):
                    idx = best_idx + offset if isinstance(best_idx, int) else None
                    if idx is None:
                        continue
                    try:
                        if 0 <= idx < len(rh) and rh[idx] not in (None, ""):
                            humidity = float(rh[idx])
                            break
                    except Exception:
                        continue

            # If still missing, log a warning for visibility
            if humidity is None:
                try:
                    print(f"Warning: humidity missing from Open-Meteo hourly data for lat={lat}, lon={lon}, selected_idx={best_idx}")
                except Exception:
                    pass

        return {"temperature": temp_c, "rainfall": rainfall, "humidity": humidity, "city": None}
    except Exception:
        return {"temperature": None, "rainfall": 0.0, "humidity": None, "city": None}
