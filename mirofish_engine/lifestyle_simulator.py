"""
Lifestyle Simulator - Generates realistic daily lifestyle inputs
Simulates patient behavior patterns: stress, diet, exercise, sleep
"""

import random
import numpy as np
from typing import Dict, Any
from datetime import datetime


class PatientLifestyleProfile:
    """Patient lifestyle characteristics"""
    
    def __init__(
        self,
        occupation: str = "office_worker",
        exercise_frequency: str = "low",  # low, moderate, high
        diet_quality: str = "poor",  # poor, moderate, good
        sleep_pattern: str = "insufficient",  # poor, insufficient, adequate, good
        stress_level: str = "moderate"  # low, moderate, high
    ):
        self.occupation = occupation
        self.exercise_frequency = exercise_frequency
        self.diet_quality = diet_quality
        self.sleep_pattern = sleep_pattern
        self.stress_level = stress_level


class LifestyleSimulator:
    """
    Simulates realistic daily lifestyle inputs for patient
    This is what was missing - agents need realistic inputs to respond to!
    """
    
    def __init__(self, profile: PatientLifestyleProfile):
        self.profile = profile
        self.day = 0
        
        # Base values from profile
        self.base_stress = self._get_base_stress()
        self.base_sleep_quality = self._get_base_sleep()
        self.base_exercise = self._get_base_exercise()
        self.base_diet_quality = self._get_base_diet()
    
    def _get_base_stress(self) -> float:
        """Get baseline stress from profile"""
        stress_map = {
            'low': 0.2,
            'moderate': 0.5,
            'high': 0.7
        }
        return stress_map.get(self.profile.stress_level, 0.5)
    
    def _get_base_sleep(self) -> float:
        """Get baseline sleep quality from profile"""
        sleep_map = {
            'poor': 0.4,
            'insufficient': 0.6,
            'adequate': 0.75,
            'good': 0.9
        }
        return sleep_map.get(self.profile.sleep_pattern, 0.6)
    
    def _get_base_exercise(self) -> float:
        """Get baseline exercise level from profile"""
        exercise_map = {
            'low': 0.2,
            'moderate': 0.5,
            'high': 0.8
        }
        return exercise_map.get(self.profile.exercise_frequency, 0.2)
    
    def _get_base_diet(self) -> float:
        """Get baseline diet quality from profile"""
        diet_map = {
            'poor': 0.3,
            'moderate': 0.6,
            'good': 0.9
        }
        return diet_map.get(self.profile.diet_quality, 0.3)
    
    def get_daily_inputs(self, day: int) -> Dict[str, Any]:
        """
        Generate realistic daily lifestyle inputs
        This is the key to making disease emerge!
        """
        self.day = day
        
        # Day of week effects
        is_weekday = day % 7 < 5
        is_monday = day % 7 == 0
        is_friday = day % 7 == 4
        is_weekend = not is_weekday
        
        # Work stress (higher on weekdays, especially Monday)
        stress = self.base_stress
        if is_weekday:
            stress += 0.15  # Work stress
            if is_monday:
                stress += 0.1  # Monday blues
        else:
            stress -= 0.1  # Weekend relaxation
        
        # Add random variation
        stress += random.uniform(-0.05, 0.05)
        stress = np.clip(stress, 0.0, 1.0)
        
        # Sleep quality (worse on weekdays, better on weekends)
        sleep_quality = self.base_sleep_quality
        if is_weekday:
            sleep_quality -= 0.1  # Work nights
        else:
            sleep_quality += 0.15  # Weekend catch-up
        sleep_quality += random.uniform(-0.1, 0.1)
        sleep_quality = np.clip(sleep_quality, 0.3, 1.0)
        
        # Exercise (mostly on weekends for sedentary people)
        exercise = self.base_exercise
        if is_weekend:
            exercise += 0.3  # Weekend activity
        if is_friday:
            exercise += 0.1  # Friday gym
        exercise += random.uniform(-0.05, 0.05)
        exercise = np.clip(exercise, 0.0, 1.0)
        
        # Diet - simulate meals
        meals = self._simulate_meals(is_weekday)
        
        return {
            'lifestyle_stress': stress,
            'sleep_quality': sleep_quality,
            'exercise': exercise,
            'food_glucose': meals['glucose_load'],
            'dietary_fat': meals['fat_grams'],
            'calories': meals['total_calories']
        }
    
    def _simulate_meals(self, is_weekday: bool) -> Dict[str, float]:
        """Simulate daily meals based on diet quality"""
        
        if self.profile.diet_quality == 'poor':
            # High carb, high fat, processed foods
            breakfast = {
                'glucose': 1.5,  # Sugary cereal, white bread
                'fat': 25,
                'calories': 600
            }
            lunch = {
                'glucose': 2.0,  # Fast food, white rice
                'fat': 35,
                'calories': 900
            }
            dinner = {
                'glucose': 2.5,  # Pasta, pizza, fried foods
                'fat': 40,
                'calories': 1000
            }
            snacks = {
                'glucose': 1.0,  # Chips, cookies, soda
                'fat': 20,
                'calories': 500
            }
            
        elif self.profile.diet_quality == 'moderate':
            # Mixed diet
            breakfast = {
                'glucose': 1.0,
                'fat': 15,
                'calories': 400
            }
            lunch = {
                'glucose': 1.5,
                'fat': 20,
                'calories': 600
            }
            dinner = {
                'glucose': 1.5,
                'fat': 25,
                'calories': 700
            }
            snacks = {
                'glucose': 0.5,
                'fat': 10,
                'calories': 200
            }
            
        else:  # good
            # Mediterranean-style, low glycemic
            breakfast = {
                'glucose': 0.5,
                'fat': 10,
                'calories': 300
            }
            lunch = {
                'glucose': 0.8,
                'fat': 15,
                'calories': 500
            }
            dinner = {
                'glucose': 1.0,
                'fat': 20,
                'calories': 600
            }
            snacks = {
                'glucose': 0.3,
                'fat': 5,
                'calories': 150
            }
        
        # Weekend vs weekday variation
        if not is_weekday:
            # Weekend: more indulgent
            for meal in [breakfast, lunch, dinner, snacks]:
                meal['glucose'] *= 1.2
                meal['fat'] *= 1.15
                meal['calories'] *= 1.1
        
        # Total daily intake
        total_glucose = (breakfast['glucose'] + lunch['glucose'] + 
                        dinner['glucose'] + snacks['glucose'])
        total_fat = (breakfast['fat'] + lunch['fat'] + 
                    dinner['fat'] + snacks['fat'])
        total_calories = (breakfast['calories'] + lunch['calories'] + 
                         dinner['calories'] + snacks['calories'])
        
        # Add random variation
        total_glucose += random.uniform(-0.3, 0.3)
        total_fat += random.uniform(-10, 10)
        total_calories += random.uniform(-100, 100)
        
        return {
            'glucose_load': max(0, total_glucose),
            'fat_grams': max(0, total_fat),
            'total_calories': max(0, total_calories)
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary of lifestyle"""
        return f"""Lifestyle Profile:
  Occupation: {self.profile.occupation}
  Exercise: {self.profile.exercise_frequency}
  Diet: {self.profile.diet_quality}
  Sleep: {self.profile.sleep_pattern}
  Stress: {self.profile.stress_level}
  
Daily Patterns:
  Base stress: {self.base_stress:.1%}
  Base sleep quality: {self.base_sleep_quality:.1%}
  Base exercise: {self.base_exercise:.1%}
  Diet quality: {self.base_diet_quality:.1%}
"""


def create_lifestyle_from_medical_report(report_data: Dict[str, Any]) -> PatientLifestyleProfile:
    """Create lifestyle profile from medical report data"""
    
    # Extract lifestyle indicators from report
    bmi = report_data.get('bmi', 25)
    exercise_sessions = report_data.get('exercise_sessions_per_week', 1)
    sleep_hours = report_data.get('sleep_hours', 6.5)
    occupation = report_data.get('occupation', 'office_worker')
    
    # Infer lifestyle quality
    if exercise_sessions < 2:
        exercise_freq = 'low'
    elif exercise_sessions < 4:
        exercise_freq = 'moderate'
    else:
        exercise_freq = 'high'
    
    if sleep_hours < 6:
        sleep_pattern = 'poor'
    elif sleep_hours < 7:
        sleep_pattern = 'insufficient'
    elif sleep_hours < 8:
        sleep_pattern = 'adequate'
    else:
        sleep_pattern = 'good'
    
    if bmi > 28:
        diet_quality = 'poor'
    elif bmi > 25:
        diet_quality = 'moderate'
    else:
        diet_quality = 'good'
    
    if occupation in ['office_worker', 'desk_job', 'sedentary']:
        stress_level = 'moderate'
    elif occupation in ['manual_labor', 'retail']:
        stress_level = 'high'
    else:
        stress_level = 'low'
    
    return PatientLifestyleProfile(
        occupation=occupation,
        exercise_frequency=exercise_freq,
        diet_quality=diet_quality,
        sleep_pattern=sleep_pattern,
        stress_level=stress_level
    )
