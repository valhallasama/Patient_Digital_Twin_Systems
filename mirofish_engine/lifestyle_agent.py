#!/usr/bin/env python3
"""
Lifestyle Agent for Multi-Agent Digital Twin
Models behavioral factors and their evolution over time
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Base state for all health agents"""
    timestamp: int = 0
    history: List[Dict] = field(default_factory=list)
    
    def record(self, data: Dict):
        """Record state to history"""
        self.history.append({
            'timestamp': self.timestamp,
            **data
        })
        self.timestamp += 1


class LifestyleAgent:
    """
    Models lifestyle behaviors and their evolution
    Influences all other organ agents through behavioral signals
    """
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        
        # Get lifestyle data
        lifestyle = patient_data.get('lifestyle', {})
        
        # Behavioral parameters (categorical)
        self.exercise_level = lifestyle.get('physical_activity', 'moderate')
        self.diet_quality = lifestyle.get('diet_quality', 'fair')
        self.sleep_duration = lifestyle.get('sleep_duration', 7.0)
        self.stress_level = lifestyle.get('stress_level', 'moderate')
        self.smoking_status = lifestyle.get('smoking_status', 'never')
        self.alcohol_consumption = lifestyle.get('alcohol_consumption', 'none')
        
        # Numeric representations (0-1 scale)
        self.exercise_numeric = self._to_numeric(self.exercise_level, 'exercise')
        self.diet_numeric = self._to_numeric(self.diet_quality, 'quality')
        self.stress_numeric = self._to_numeric(self.stress_level, 'stress')
        self.smoking_numeric = 1.0 if self.smoking_status == 'current' else 0.0
        self.alcohol_numeric = self._to_numeric(self.alcohol_consumption, 'alcohol')
        self.sleep_quality = self.sleep_duration / 8.0  # 0-1 scale
        
        # Behavioral change parameters
        self.motivation = 0.5  # Willingness to change (0-1)
        self.adherence = 0.7  # Adherence to interventions (0-1)
        self.social_support = 0.5  # Social support level (0-1)
        
        # Environmental factors
        self.work_stress = self._to_numeric(self.stress_level, 'stress')
        self.financial_stress = 0.3
        self.access_to_healthy_food = 0.7
        self.access_to_exercise = 0.7
        
        # Store last perception
        self.last_perception = {}
        
        # Patient demographics for context
        self.age = patient_data.get('age', 40)
        self.occupation = patient_data.get('occupation', 'unknown')
    
    def _to_numeric(self, value: str, category: str) -> float:
        """Convert categorical to numeric"""
        mappings = {
            'exercise': {
                'sedentary': 0.0, 'light': 0.3, 'moderate': 0.6, 'vigorous': 1.0
            },
            'quality': {
                'poor': 0.2, 'fair': 0.5, 'good': 0.8, 'excellent': 1.0
            },
            'stress': {
                'low': 0.2, 'moderate': 0.5, 'high': 0.9
            },
            'alcohol': {
                'none': 0.0, 'light': 0.3, 'moderate': 0.6, 'heavy': 1.0
            }
        }
        return mappings.get(category, {}).get(value.lower(), 0.5)
    
    def _from_numeric(self, value: float, category: str) -> str:
        """Convert numeric back to categorical"""
        mappings = {
            'exercise': [(0.0, 'sedentary'), (0.3, 'light'), (0.6, 'moderate'), (1.0, 'vigorous')],
            'quality': [(0.2, 'poor'), (0.5, 'fair'), (0.8, 'good'), (1.0, 'excellent')],
            'stress': [(0.2, 'low'), (0.5, 'moderate'), (0.9, 'high')],
            'alcohol': [(0.0, 'none'), (0.3, 'light'), (0.6, 'moderate'), (1.0, 'heavy')]
        }
        
        thresholds = mappings.get(category, [])
        for threshold, label in reversed(thresholds):
            if value >= threshold:
                return label
        return thresholds[0][1] if thresholds else 'unknown'
    
    def perceive(self, signals: Dict):
        """Receive signals from other agents and environment"""
        self.last_perception = signals
        
        # Health feedback influences behavior
        # Poor health outcomes may increase motivation to change
        metabolic_stress = signals.get('metabolic_stress', 0.0)
        if metabolic_stress > 0.5:
            self.motivation += 0.01  # Gradual increase in motivation
            self.motivation = min(self.motivation, 1.0)
        
        # Cardiovascular stress
        vascular_stress = signals.get('vascular_stress', 0.0)
        if vascular_stress > 0.3:
            self.motivation += 0.005
            self.motivation = min(self.motivation, 1.0)
        
        # Intervention signals (from scenario simulator)
        if 'intervention' in signals:
            intervention = signals['intervention']
            
            # Apply intervention effects with adherence
            if 'exercise_target' in intervention:
                target = self._to_numeric(intervention['exercise_target'], 'exercise')
                self.exercise_numeric += (target - self.exercise_numeric) * self.adherence * 0.1
            
            if 'diet_target' in intervention:
                target = self._to_numeric(intervention['diet_target'], 'quality')
                self.diet_numeric += (target - self.diet_numeric) * self.adherence * 0.1
            
            if 'stress_reduction' in intervention:
                self.stress_numeric *= (1 - intervention['stress_reduction'] * self.adherence * 0.1)
    
    def act(self) -> Dict:
        """Update lifestyle state and emit signals"""
        
        # Natural behavioral drift (regression to baseline)
        # Without intervention, behaviors tend to worsen slightly
        drift_rate = 0.001
        
        # Exercise tends to decrease with age and stress
        self.exercise_numeric *= (1 - drift_rate)
        if self.work_stress > 0.7:
            self.exercise_numeric *= 0.999
        
        # Diet quality affected by stress and access
        if self.work_stress > 0.7:
            self.diet_numeric *= 0.9995
        if self.access_to_healthy_food < 0.5:
            self.diet_numeric *= 0.999
        
        # Stress accumulation
        self.stress_numeric += np.random.normal(0, 0.01)
        self.stress_numeric = max(0, min(self.stress_numeric, 1.0))
        
        # Sleep affected by stress
        if self.stress_numeric > 0.7:
            self.sleep_duration -= 0.01
            self.sleep_duration = max(5.0, self.sleep_duration)
        
        # Motivation-driven improvement
        if self.motivation > 0.6:
            # High motivation leads to gradual improvement
            improvement_rate = (self.motivation - 0.5) * 0.01
            
            self.exercise_numeric += improvement_rate
            self.diet_numeric += improvement_rate
            self.stress_numeric -= improvement_rate * 0.5
            
            # Bounds
            self.exercise_numeric = min(self.exercise_numeric, 1.0)
            self.diet_numeric = min(self.diet_numeric, 1.0)
            self.stress_numeric = max(0, self.stress_numeric)
        
        # Smoking cessation (if motivated)
        if self.smoking_status == 'current' and self.motivation > 0.8:
            # Small chance of quitting
            if np.random.random() < 0.001:
                self.smoking_status = 'former'
                self.smoking_numeric = 0.0
        
        # Update categorical from numeric
        self.exercise_level = self._from_numeric(self.exercise_numeric, 'exercise')
        self.diet_quality = self._from_numeric(self.diet_numeric, 'quality')
        self.stress_level = self._from_numeric(self.stress_numeric, 'stress')
        self.alcohol_consumption = self._from_numeric(self.alcohol_numeric, 'alcohol')
        
        # Record state
        self.state.record({
            'exercise_level': self.exercise_level,
            'exercise_numeric': self.exercise_numeric,
            'diet_quality': self.diet_quality,
            'diet_numeric': self.diet_numeric,
            'stress_level': self.stress_level,
            'stress_numeric': self.stress_numeric,
            'smoking_status': self.smoking_status,
            'alcohol_consumption': self.alcohol_consumption,
            'sleep_duration': self.sleep_duration,
            'motivation': self.motivation,
            'adherence': self.adherence
        })
        
        # Emit signals to other agents
        return {
            'exercise_level': self.exercise_numeric,
            'diet_quality': self.diet_numeric,
            'stress_level': self.stress_numeric,
            'smoking': self.smoking_numeric,
            'alcohol': self.alcohol_numeric,
            'sleep_quality': self.sleep_quality
        }
    
    def predict_disease(self) -> List[Dict]:
        """Predict lifestyle-related health risks"""
        predictions = []
        
        # Mental health risk (stress, sleep)
        if self.stress_numeric > 0.7 or self.sleep_duration < 6:
            mental_health_risk = min(0.9, self.stress_numeric * 0.8 + (1 - self.sleep_quality) * 0.5)
            predictions.append({
                'disease': 'Mental Health Issues',
                'probability': mental_health_risk,
                'time_to_onset_years': 1.0 if mental_health_risk > 0.7 else 3.0,
                'confidence': 0.70,
                'risk_factors': [
                    f"High stress: {self.stress_level}",
                    f"Poor sleep: {self.sleep_duration:.1f}h"
                ]
            })
        
        # Substance abuse risk
        if self.smoking_numeric > 0 or self.alcohol_numeric > 0.6:
            substance_risk = max(self.smoking_numeric, self.alcohol_numeric * 0.7)
            predictions.append({
                'disease': 'Substance-Related Health Issues',
                'probability': substance_risk,
                'time_to_onset_years': 5.0,
                'confidence': 0.65,
                'risk_factors': [
                    f"Smoking: {self.smoking_status}",
                    f"Alcohol: {self.alcohol_consumption}"
                ]
            })
        
        # Sedentary lifestyle complications
        if self.exercise_numeric < 0.3:
            sedentary_risk = (0.3 - self.exercise_numeric) * 2
            predictions.append({
                'disease': 'Sedentary Lifestyle Complications',
                'probability': sedentary_risk,
                'time_to_onset_years': 3.0,
                'confidence': 0.75,
                'risk_factors': [
                    f"Low activity: {self.exercise_level}",
                    f"Motivation: {self.motivation:.2f}"
                ]
            })
        
        return predictions if predictions else [{
            'disease': 'No significant lifestyle risks',
            'probability': 0.0,
            'time_to_onset_years': 10.0,
            'confidence': 0.80,
            'risk_factors': []
        }]
    
    def apply_intervention(self, intervention_type: str, intensity: float = 0.5):
        """
        Apply behavioral intervention
        
        Args:
            intervention_type: Type of intervention
            intensity: Intervention intensity (0-1)
        """
        if intervention_type == 'exercise_program':
            self.exercise_numeric += 0.2 * intensity * self.adherence
            self.motivation += 0.1 * intensity
        
        elif intervention_type == 'dietary_counseling':
            self.diet_numeric += 0.2 * intensity * self.adherence
            self.motivation += 0.1 * intensity
        
        elif intervention_type == 'stress_management':
            self.stress_numeric -= 0.2 * intensity * self.adherence
            self.sleep_duration += 0.5 * intensity
        
        elif intervention_type == 'smoking_cessation':
            if self.smoking_status == 'current':
                success_rate = 0.3 * intensity * self.adherence
                if np.random.random() < success_rate:
                    self.smoking_status = 'former'
                    self.smoking_numeric = 0.0
        
        # Bounds
        self.exercise_numeric = max(0, min(self.exercise_numeric, 1.0))
        self.diet_numeric = max(0, min(self.diet_numeric, 1.0))
        self.stress_numeric = max(0, min(self.stress_numeric, 1.0))
        self.motivation = max(0, min(self.motivation, 1.0))
        self.sleep_duration = max(4, min(self.sleep_duration, 10))
