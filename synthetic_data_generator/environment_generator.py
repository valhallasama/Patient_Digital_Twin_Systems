import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
    def generate_air_quality_data(self, location: str, days: int = 365) -> pd.DataFrame:
        urban_factor = 1.0
        if 'urban' in location.lower() or 'city' in location.lower():
            urban_factor = 2.0
        elif 'rural' in location.lower():
            urban_factor = 0.5
        
        air_quality = []
        
        for day in range(days):
            season_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day / 365)
            
            pm25 = max(0, urban_factor * 15 * season_factor + self.rng.normal(0, 10))
            pm10 = pm25 * self.rng.uniform(1.5, 2.5)
            
            air_quality.append({
                'location': location,
                'day': day,
                'date': (datetime.now() - timedelta(days=365-day)).strftime('%Y-%m-%d'),
                'pm25_ug_m3': pm25,
                'pm10_ug_m3': pm10,
                'no2_ug_m3': max(0, urban_factor * 30 * season_factor + self.rng.normal(0, 15)),
                'o3_ug_m3': max(0, 60 + 40 * np.sin(2 * np.pi * day / 365) + self.rng.normal(0, 20)),
                'co_mg_m3': max(0, urban_factor * 0.5 + self.rng.normal(0, 0.2)),
                'so2_ug_m3': max(0, urban_factor * 10 + self.rng.normal(0, 5)),
                'aqi': self._calculate_aqi(pm25, pm10)
            })
        
        return pd.DataFrame(air_quality)
    
    def _calculate_aqi(self, pm25: float, pm10: float) -> int:
        aqi_pm25 = pm25 * 4
        aqi_pm10 = pm10 * 2
        return int(max(aqi_pm25, aqi_pm10))
    
    def generate_climate_data(self, location: str, days: int = 365) -> pd.DataFrame:
        latitude_factor = self.rng.uniform(-1, 1)
        
        climate = []
        
        for day in range(days):
            seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day - 80) / 365)
            temp_c = seasonal_temp + latitude_factor * 10 + self.rng.normal(0, 5)
            
            humidity = 60 + 20 * np.sin(2 * np.pi * day / 365) + self.rng.normal(0, 10)
            
            precipitation = max(0, self.rng.exponential(2) if self.rng.random() < 0.3 else 0)
            
            climate.append({
                'location': location,
                'day': day,
                'date': (datetime.now() - timedelta(days=365-day)).strftime('%Y-%m-%d'),
                'temperature_c': temp_c,
                'humidity_percent': np.clip(humidity, 20, 100),
                'precipitation_mm': precipitation,
                'wind_speed_kmh': max(0, self.rng.exponential(10)),
                'uv_index': max(0, 5 + 5 * np.sin(2 * np.pi * (day - 80) / 365) + self.rng.normal(0, 2)),
                'pressure_hpa': 1013 + self.rng.normal(0, 10)
            })
        
        return pd.DataFrame(climate)
    
    def generate_noise_pollution(self, location: str, days: int = 365) -> pd.DataFrame:
        urban_factor = 1.0
        if 'urban' in location.lower() or 'city' in location.lower():
            urban_factor = 1.5
        elif 'rural' in location.lower():
            urban_factor = 0.5
        
        noise_data = []
        
        for day in range(days):
            is_weekend = day % 7 in [5, 6]
            
            daytime_noise = 50 + urban_factor * 20 + self.rng.normal(0, 10)
            if is_weekend:
                daytime_noise -= 5
            
            nighttime_noise = daytime_noise - 15 + self.rng.normal(0, 5)
            
            noise_data.append({
                'location': location,
                'day': day,
                'daytime_noise_db': np.clip(daytime_noise, 30, 90),
                'nighttime_noise_db': np.clip(nighttime_noise, 25, 70),
                'peak_noise_db': np.clip(daytime_noise + self.rng.uniform(10, 30), 40, 110)
            })
        
        return pd.DataFrame(noise_data)
    
    def generate_socioeconomic_factors(self, location: str) -> Dict:
        urban_type = 'urban' if 'urban' in location.lower() or 'city' in location.lower() else 'suburban'
        if 'rural' in location.lower():
            urban_type = 'rural'
        
        if urban_type == 'urban':
            median_income = self.rng.uniform(50000, 100000)
            healthcare_access_score = self.rng.uniform(7, 10)
            education_score = self.rng.uniform(7, 10)
            crime_rate = self.rng.uniform(30, 80)
        elif urban_type == 'suburban':
            median_income = self.rng.uniform(60000, 120000)
            healthcare_access_score = self.rng.uniform(8, 10)
            education_score = self.rng.uniform(8, 10)
            crime_rate = self.rng.uniform(10, 40)
        else:
            median_income = self.rng.uniform(30000, 60000)
            healthcare_access_score = self.rng.uniform(5, 8)
            education_score = self.rng.uniform(5, 8)
            crime_rate = self.rng.uniform(5, 30)
        
        return {
            'location': location,
            'urban_type': urban_type,
            'median_income_usd': median_income,
            'poverty_rate_percent': max(0, self.rng.uniform(5, 25)),
            'unemployment_rate_percent': max(0, self.rng.uniform(3, 12)),
            'healthcare_access_score': healthcare_access_score,
            'education_score': education_score,
            'crime_rate_per_1000': crime_rate,
            'walkability_score': self.rng.uniform(3, 10),
            'green_space_percent': self.rng.uniform(10, 40),
            'food_desert': self.rng.choice([True, False], p=[0.2, 0.8])
        }
    
    def generate_complete_environment_profile(self, location: str, 
                                             days: int = 365) -> Dict[str, pd.DataFrame]:
        logger.info(f"Generating environmental data for {location}...")
        
        air_quality = self.generate_air_quality_data(location, days)
        climate = self.generate_climate_data(location, days)
        noise = self.generate_noise_pollution(location, days)
        socioeconomic = self.generate_socioeconomic_factors(location)
        
        combined_daily = air_quality.merge(climate, on=['location', 'day', 'date']) \
                                    .merge(noise, on=['location', 'day'])
        
        return {
            'air_quality': air_quality,
            'climate': climate,
            'noise': noise,
            'socioeconomic': pd.DataFrame([socioeconomic]),
            'combined_daily': combined_daily
        }


if __name__ == "__main__":
    env_gen = EnvironmentGenerator()
    
    locations = ['New York City Urban', 'Rural Vermont', 'Suburban California']
    
    for location in locations:
        env_data = env_gen.generate_complete_environment_profile(location, days=30)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Location: {location}")
        logger.info(f"{'='*60}")
        logger.info(f"\nAir Quality Summary:\n{env_data['air_quality'].describe()}")
        logger.info(f"\nSocioeconomic Factors:\n{env_data['socioeconomic'].to_dict('records')[0]}")
