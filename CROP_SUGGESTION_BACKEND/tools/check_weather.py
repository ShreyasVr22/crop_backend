import sys
from pathlib import Path
# Ensure package root is on sys.path so `from app...` imports work when run from different cwd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.weather import fetch_current_weather

coords = [
    ("Bangalore", 12.9716, 77.5946),
    ("Mysuru", 12.2958, 76.6394),
    ("Belagavi", 15.8497, 74.4977),
]

for name, lat, lon in coords:
    now = fetch_current_weather(lat, lon, target_hour=None)
    print(f"{name} NOW -> temp={now.get('temperature')}, humidity={now.get('humidity')}, rainfall={now.get('rainfall')}")
