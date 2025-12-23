import json
from pathlib import Path
from typing import List, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CROPS_DB_PATH = BASE_DIR / "data" / "crops_detailed.json"

class CropRecommender:
    def __init__(self, db_path=CROPS_DB_PATH):
        with open(db_path, "r", encoding="utf-8") as f:
            self.crops_db = json.load(f)
        self.crops = self.crops_db["crops"]

    def _score_npk(self, n: float, p: float, k: float, crop: Dict) -> float:
        """Score NPK levels: 0-1"""
        soil_req = crop["soil_requirements"]
        
        n_score = self._score_range(n, soil_req["min_nitrogen"], soil_req["max_nitrogen"])
        p_score = self._score_range(p, soil_req["min_phosphorus"], soil_req["max_phosphorus"])
        k_score = self._score_range(k, soil_req["min_potassium"], soil_req["max_potassium"])
        
        return (n_score * 0.4 + p_score * 0.3 + k_score * 0.3)

    def _score_ph(self, ph: float, crop: Dict) -> float:
        """Score pH: 0-1"""
        soil_req = crop["soil_requirements"]
        return self._score_range(ph, soil_req["min_ph"], soil_req["max_ph"])

    def _score_soil_type(self, soil_type: str, crop: Dict) -> float:
        """Score soil type: exact match=1.0, partial=0.6, no match=0.3"""
        allowed = [s.lower() for s in crop["soil_requirements"]["soil_types"]]
        soil_type_lower = soil_type.lower()
        
        if soil_type_lower in allowed:
            return 1.0
        if any(soil_type_lower in s or s in soil_type_lower for s in allowed):
            return 0.6
        return 0.3

    def _score_soil_moisture(self, moisture: str, crop: Dict) -> float:
        """Score moisture: exact match=1.0, no match=0.4"""
        allowed = [m.lower() for m in crop["soil_requirements"]["moisture_types"]]
        return 1.0 if moisture.lower() in allowed else 0.4

    def _score_temperature(self, temp: float, crop: Dict) -> float:
        """Score temperature: 0-1"""
        clim_req = crop["climate_requirements"]
        return self._score_range(temp, clim_req["min_temperature"], clim_req["max_temperature"])

    def _score_rainfall(self, rainfall: float, crop: Dict) -> float:
        """Score rainfall: 0-1"""
        clim_req = crop["climate_requirements"]
        min_rain = clim_req["min_rainfall"]
        # No max, but penalize very high rainfall
        if rainfall < min_rain:
            return max(0, rainfall / min_rain * 0.5)
        if rainfall > min_rain * 3:
            return 0.7  # too much rain
        return 1.0

    def _score_humidity(self, humidity: float, crop: Dict) -> float:
        """Score humidity: 0-1"""
        clim_req = crop["climate_requirements"]
        min_h = clim_req["min_humidity"]
        max_h = clim_req["max_humidity"]
        return self._score_range(humidity, min_h, max_h)

    def _score_season(self, season: str, crop: Dict) -> float:
        """Score season: exact match=1.0, no match=0.2"""
        allowed = [s.lower() for s in crop["climate_requirements"]["seasons"]]
        return 1.0 if season.lower() in allowed else 0.2

    def _score_region(self, region: str, crop: Dict) -> float:
        """Score region: best=1.0, suitable=0.8, other=0.4"""
        region_lower = region.lower()
        best = [r.lower() for r in crop["regional_suitability"].get("best_regions", [])]
        suitable = [r.lower() for r in crop["regional_suitability"].get("suitable_regions", [])]
        
        if any(region_lower in b or b in region_lower for b in best):
            return 1.0
        if any(region_lower in s or s in region_lower for s in suitable):
            return 0.8
        return 0.4

    def _score_range(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value within range to 0-1 score"""
        if value < min_val:
            deficit = min_val - value
            return max(0, 1 - (deficit / min_val * 0.5))
        if value > max_val:
            excess = value - max_val
            return max(0, 1 - (excess / max_val * 0.3))
        return 1.0

    def recommend(
        self,
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
        region: str,
    ) -> List[Dict]:
        """
        Comprehensive crop recommendation considering ALL factors.
        
        Weights:
        - Soil factors (NPK + pH + type + moisture): 45%
        - Climate factors (temp + rainfall + humidity + season): 40%
        - Regional suitability: 15%
        """
        
        recommendations = []
        
        for crop in self.crops:
            # Soil score (45%)
            npk_score = self._score_npk(nitrogen, phosphorus, potassium, crop)
            ph_score = self._score_ph(ph_level, crop)
            soil_type_score = self._score_soil_type(soil_type, crop)
            moisture_score = self._score_soil_moisture(soil_moisture, crop)
            soil_score = (
                npk_score * 0.4 +
                ph_score * 0.3 +
                soil_type_score * 0.2 +
                moisture_score * 0.1
            )
            
            # Climate score (40%)
            temp_score = self._score_temperature(temperature, crop)
            rain_score = self._score_rainfall(rainfall, crop)
            humidity_score = self._score_humidity(humidity, crop)
            season_score = self._score_season(season, crop)
            climate_score = (
                temp_score * 0.35 +
                rain_score * 0.25 +
                humidity_score * 0.25 +
                season_score * 0.15
            )
            
            # Region score (15%)
            region_score = self._score_region(region, crop)
            
            # Final weighted score
            final_score = (
                soil_score * 0.45 +
                climate_score * 0.40 +
                region_score * 0.15
            )
            
            if final_score >= 0.4:  # Include crops with reasonable suitability
                recommendations.append({
                    "crop_id": crop["id"],
                    "name": crop["name"],
                    "kannada_name": crop.get("kannada_name", ""),
                    "score": round(final_score, 3),
                    "soil_score": round(soil_score, 3),
                    "climate_score": round(climate_score, 3),
                    "region_score": round(region_score, 3),
                    "yield_potential": crop["yield_potential"],
                })
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:10]  # Top 10

    def get_crop_details(self, crop_id: str) -> Dict:
        """Get full details for a specific crop"""
        for crop in self.crops:
            if crop["id"] == crop_id:
                return crop
        return None
