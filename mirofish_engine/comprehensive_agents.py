#!/usr/bin/env python3
"""
Comprehensive Health Agents for Medical Digital Twin
MiroFish-style autonomous agents with medical theory-based simulation
"""

import numpy as np
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass, field
import pickle
from pathlib import Path

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


class MetabolicAgent:
    """
    Manages glucose, insulin, energy metabolism
    Predicts: Type 2 Diabetes, Metabolic Syndrome
    Uses: Rule-based simulation + ML model calibration
    """
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        
        # Core parameters (with medical theory-based defaults)
        self.glucose = patient_data.get('fasting_glucose', self._estimate_glucose(patient_data))
        self.hba1c = patient_data.get('hba1c', self._estimate_hba1c(self.glucose))
        self.insulin = patient_data.get('insulin', self._estimate_insulin(patient_data))
        self.bmi = patient_data.get('bmi', self._calculate_bmi(patient_data))
        
        # Internal state
        self.insulin_sensitivity = self._calculate_insulin_sensitivity()
        self.beta_cell_function = 1.0  # 100% function initially
        self.metabolic_age = patient_data.get('age', 40)
        
        # Risk factors
        self.family_history_diabetes = patient_data.get('family_history', {}).get('diabetes', False)
        self.lifestyle_score = self._calculate_lifestyle_score(patient_data)
        
        # Store last perception
        self.last_perception = {}
        
        # Load ML model for calibration (if available)
        self.ml_model = self._load_ml_model()
        self.patient_data = patient_data  # Store for ML features
    
    def _estimate_glucose(self, data: Dict) -> float:
        """Estimate fasting glucose from available data using medical theory"""
        # If HbA1c available, estimate glucose
        if 'hba1c' in data:
            # Formula: Average glucose (mg/dL) ≈ (HbA1c × 28.7) - 46.7
            return (data['hba1c'] * 28.7) - 46.7
        
        # Age-based estimation (healthy baseline)
        age = data.get('age', 40)
        base_glucose = 90 + (age - 30) * 0.1  # Slight increase with age
        
        # Adjust for BMI
        bmi = data.get('bmi', self._calculate_bmi(data))
        if bmi > 25:
            base_glucose += (bmi - 25) * 1.5
        
        return max(70, min(base_glucose, 125))  # Normal range
    
    def _estimate_hba1c(self, glucose: float) -> float:
        """Estimate HbA1c from glucose"""
        # Formula: HbA1c ≈ (Average glucose + 46.7) / 28.7
        return (glucose + 46.7) / 28.7
    
    def _estimate_insulin(self, data: Dict) -> float:
        """Estimate fasting insulin"""
        # Normal range: 2-20 µU/mL
        # Higher with obesity, insulin resistance
        bmi = data.get('bmi', self._calculate_bmi(data))
        
        if bmi < 25:
            return 5.0  # Normal
        elif bmi < 30:
            return 10.0  # Slightly elevated
        else:
            return 15.0  # Elevated
    
    def _calculate_bmi(self, data: Dict) -> float:
        """Calculate BMI from height and weight"""
        if 'bmi' in data:
            return data['bmi']
        
        if 'height' in data and 'weight' in data:
            height_m = data['height'] / 100  # cm to m
            return data['weight'] / (height_m ** 2)
        
        # Default to healthy BMI
        return 22.0
    
    def _calculate_insulin_sensitivity(self) -> float:
        """Calculate insulin sensitivity (HOMA-IR based)"""
        # HOMA-IR = (Glucose × Insulin) / 405
        # Lower is better, <2.5 is normal
        homa_ir = (self.glucose * self.insulin) / 405
        
        # Convert to sensitivity (inverse)
        sensitivity = 1.0 / (1.0 + homa_ir / 2.5)
        return sensitivity
    
    def _calculate_lifestyle_score(self, data: Dict) -> float:
        """Calculate lifestyle quality score (0-1)"""
        score = 0.5  # Neutral baseline
        
        lifestyle = data.get('lifestyle', {})
        
        # Physical activity
        activity = lifestyle.get('physical_activity', 'moderate')
        if activity == 'vigorous':
            score += 0.2
        elif activity == 'moderate':
            score += 0.1
        elif activity == 'sedentary':
            score -= 0.2
        
        # Diet quality
        diet = lifestyle.get('diet_quality', 'fair')
        if diet == 'excellent':
            score += 0.2
        elif diet == 'good':
            score += 0.1
        elif diet == 'poor':
            score -= 0.2
        
        # Smoking
        if lifestyle.get('smoking_status') == 'current':
            score -= 0.3
        
        return max(0, min(score, 1.0))
    
    def perceive(self, signals: Dict):
        """Receive signals from other agents and environment"""
        # Store for use in act()
        self.last_perception = signals
        
        # Cardiovascular signals
        if 'blood_pressure' in signals:
            bp = signals['blood_pressure']
            if bp['systolic'] > 140:
                self.insulin_sensitivity *= 0.98  # Hypertension reduces sensitivity
        
        # Hepatic signals
        if 'liver_fat' in signals:
            if signals['liver_fat'] > 0.3:
                self.insulin_sensitivity *= 0.95  # Fatty liver reduces sensitivity
        
        # Lifestyle signals
        if 'exercise' in signals:
            self.insulin_sensitivity *= (1.0 + signals['exercise'] * 0.02)
        
        if 'stress' in signals:
            self.glucose += signals['stress'] * 5  # Stress raises glucose
    
    def act(self) -> Dict:
        """Update metabolic state based on perceptions and lifestyle"""
        # Get lifestyle factors from perceptions
        exercise = self.last_perception.get('exercise_level', 0.5)
        diet = self.last_perception.get('diet_quality', 0.5)
        stress = self.last_perception.get('stress_level', 0.5)
        smoking = self.last_perception.get('smoking', 0.0)
        
        # Calculate daily glucose change based on lifestyle
        glucose_change = 0.0
        
        # Poor diet increases glucose
        if diet < 0.5:
            glucose_change += (0.5 - diet) * 0.3  # Up to +0.15 mg/dL per day
        
        # Exercise decreases glucose
        if exercise > 0.3:
            glucose_change -= exercise * 0.2  # Up to -0.2 mg/dL per day
        
        # Stress increases glucose
        glucose_change += stress * 0.1
        
        # ML model calibration: adjust progression rate based on patient risk profile
        ml_adjustment = self._get_ml_risk_adjustment()
        glucose_change *= ml_adjustment
        
        # Smoking increases insulin resistance
        if smoking > 0:
            self.insulin_sensitivity *= 0.9998  # Gradual decline
        
        # Age-related decline in insulin sensitivity
        self.insulin_sensitivity *= 0.99995
        
        # Apply glucose change
        self.glucose += glucose_change
        self.glucose += np.random.normal(0, 1)  # Daily variation
        self.glucose = max(70, min(self.glucose, 300))  # Physiological bounds
        
        # Beta cell deterioration under chronic stress
        if self.insulin_sensitivity < 0.7 or self.glucose > 140:
            self.beta_cell_function *= 0.9997  # Faster decline under stress
        
        # Update HbA1c (3-month weighted average)
        # HbA1c changes slowly - weighted average of recent glucose
        target_hba1c = self._estimate_hba1c(self.glucose)
        self.hba1c = self.hba1c * 0.99 + target_hba1c * 0.01  # Slow convergence
        
        # Record state
        self.state.record({
            'glucose': self.glucose,
            'hba1c': self.hba1c,
            'insulin_sensitivity': self.insulin_sensitivity,
            'beta_cell_function': self.beta_cell_function,
            'lifestyle_impact': {
                'exercise': exercise,
                'diet': diet,
                'stress': stress
            }
        })
        
        return {
            'glucose_level': self.glucose / 126,  # Normalized
            'metabolic_stress': 1.0 - self.insulin_sensitivity
        }
    
    def predict_disease(self) -> Dict:
        """Predict diabetes risk based on current trajectory"""
        # Check if already diabetic
        if self.hba1c >= 6.5:
            return {
                'disease': 'Type 2 Diabetes',
                'probability': 1.0,
                'time_to_onset_years': 0.0,
                'time_to_onset_days': 0,
                'confidence': 0.95,
                'status': 'CURRENT DIAGNOSIS',
                'current_hba1c': round(self.hba1c, 2),
                'risk_factors': self._get_risk_factors()
            }
        
        # Estimate progression rate based on current trajectory
        # Look at recent history to calculate rate of change
        if len(self.state.history) > 30:
            # Get HbA1c from 30 days ago
            old_hba1c = self.state.history[-30]['hba1c']
            hba1c_change_per_day = (self.hba1c - old_hba1c) / 30
        else:
            # Estimate based on lifestyle
            lifestyle = self.last_perception
            exercise = lifestyle.get('exercise_level', 0.5)
            diet = lifestyle.get('diet_quality', 0.5)
            
            # Poor lifestyle → faster progression
            if diet < 0.5 and exercise < 0.3:
                hba1c_change_per_day = 0.003  # ~1% per year
            elif diet < 0.6 or exercise < 0.4:
                hba1c_change_per_day = 0.0015  # ~0.5% per year
            else:
                hba1c_change_per_day = 0.0005  # ~0.2% per year
        
        # Calculate days until HbA1c reaches 6.5%
        if hba1c_change_per_day > 0:
            days_to_diabetes = (6.5 - self.hba1c) / hba1c_change_per_day
            years_to_diabetes = days_to_diabetes / 365
        else:
            # Improving or stable
            days_to_diabetes = float('inf')
            years_to_diabetes = 10.0
        
        # Calculate probability based on trajectory
        if self.hba1c >= 6.0:
            probability = 0.85
        elif self.hba1c >= 5.7:
            probability = 0.50 + (self.hba1c - 5.7) * 0.5
        else:
            probability = max(0.05, (self.hba1c - 4.5) * 0.2)
        
        # Adjust for other risk factors
        if self.bmi > 30:
            probability += 0.1
        if self.family_history_diabetes:
            probability += 0.1
        if self.insulin_sensitivity < 0.5:
            probability += 0.15
        
        probability = min(probability, 0.95)
        
        return {
            'disease': 'Type 2 Diabetes',
            'probability': probability,
            'time_to_onset_years': min(years_to_diabetes, 10.0),
            'time_to_onset_days': int(min(days_to_diabetes, 3650)),
            'confidence': 0.80,
            'status': 'FUTURE RISK',
            'current_hba1c': round(self.hba1c, 2),
            'projected_hba1c': round(min(self.hba1c + hba1c_change_per_day * 365, 10.0), 2),
            'progression_rate': f"{hba1c_change_per_day * 365:.3f}% per year",
            'risk_factors': self._get_risk_factors()
        }
    
    def _load_ml_model(self):
        """Load trained ML model for metabolic predictions"""
        try:
            model_path = Path(__file__).parent.parent / 'models' / 'trained' / 'metabolic_model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load ML model: {e}")
        return None
    
    def _get_ml_risk_adjustment(self) -> float:
        """Use ML model to adjust progression rate"""
        if self.ml_model is None:
            return 1.0  # No adjustment
        
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features()
            
            # Get ML prediction probability
            ml_risk = self.ml_model.predict_proba([features])[0][1]  # Probability of diabetes
            
            # Convert to progression rate adjustment (0.5 to 1.5x)
            # Higher ML risk → faster progression
            adjustment = 0.5 + ml_risk
            return adjustment
        except Exception as e:
            print(f"Warning: ML prediction failed: {e}")
            return 1.0
    
    def _prepare_ml_features(self) -> List:
        """Prepare features for ML model"""
        # Match training data features
        return [
            self.metabolic_age,
            self.bmi,
            self.hba1c,
            self.glucose,
            1 if self.family_history_diabetes else 0,
            self.insulin_sensitivity,
            self.lifestyle_score
        ]
    
    def _get_risk_factors(self) -> List[str]:
        """Identify active risk factors"""
        factors = []
        
        if self.hba1c >= 5.7:
            factors.append(f"HbA1c {self.hba1c:.1f}% (prediabetes)")
        if self.bmi > 25:
            factors.append(f"BMI {self.bmi:.1f} (overweight/obese)")
        if self.family_history_diabetes:
            factors.append("Family history of diabetes")
        if self.insulin_sensitivity < 0.6:
            factors.append("Insulin resistance detected")
        if self.lifestyle_score < 0.4:
            factors.append("Poor lifestyle factors")
        
        return factors


class CardiovascularAgent:
    """
    Manages heart, blood vessels, circulation
    Predicts: Heart Attack, Stroke, Hypertension
    """
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        
        # Core parameters
        self.systolic_bp = patient_data.get('blood_pressure', {}).get('systolic', 120)
        self.diastolic_bp = patient_data.get('blood_pressure', {}).get('diastolic', 80)
        self.total_cholesterol = patient_data.get('total_cholesterol', 180)
        self.ldl = patient_data.get('ldl_cholesterol', 100)
        self.hdl = patient_data.get('hdl_cholesterol', 50)
        self.triglycerides = patient_data.get('triglycerides', 150)
        
        # Internal state
        self.vessel_elasticity = 1.0
        self.atherosclerosis_level = 0.0
        self.age = patient_data.get('age', 40)
        self.endothelial_function = 1.0  # 0-1 scale
        
        # Risk factors
        self.family_history_cvd = patient_data.get('family_history', {}).get('cardiovascular_disease', False)
        
        # Store last perception
        self.last_perception = {}
        
        # Sex and smoking status
        self.sex = patient_data.get('sex', 'M')
        self.smoking = patient_data.get('lifestyle', {}).get('smoking_status') == 'current'
    
    def _estimate_bp(self, data: Dict) -> tuple:
        """Estimate blood pressure from age and risk factors"""
        age = data.get('age', 40)
        
        # Age-based baseline
        systolic = 110 + (age - 30) * 0.5
        diastolic = 70 + (age - 30) * 0.2
        
        # Adjust for BMI
        bmi = data.get('bmi', 22)
        if bmi > 25:
            systolic += (bmi - 25) * 1.5
            diastolic += (bmi - 25) * 0.5
        
        return (systolic, diastolic)
    
    def perceive(self, signals: Dict):
        """Receive signals from other agents"""
        # Store for use in act()
        self.last_perception = signals
        
        # Metabolic signals affect cardiovascular health
        if 'glucose_level' in signals:
            if signals['glucose_level'] > 1.2:  # High glucose
                self.atherosclerosis_level += 0.001
        
        if 'metabolic_stress' in signals:
            self.systolic_bp += signals['metabolic_stress'] * 2
        
        # Inflammation signals
        if 'inflammation_level' in signals:
            self.atherosclerosis_level += signals['inflammation_level'] * 0.001
        
        # Stress signals
        if 'stress' in signals:
            self.systolic_bp += signals['stress'] * 2
    
    def act(self):
        """Update cardiovascular state based on lifestyle and metabolic signals"""
        # Get lifestyle factors
        exercise = self.last_perception.get('exercise_level', 0.5)
        diet = self.last_perception.get('diet_quality', 0.5)
        stress = self.last_perception.get('stress_level', 0.5)
        smoking = self.last_perception.get('smoking', 0.0)
        
        # Get metabolic signals
        glucose = self.last_perception.get('glucose_level', 0.7)
        metabolic_stress = self.last_perception.get('metabolic_stress', 0.3)
        
        # === BLOOD PRESSURE EVOLUTION ===
        bp_change = 0.0
        
        # Exercise lowers BP
        if exercise > 0.5:
            bp_change -= (exercise - 0.5) * 0.15  # Up to -0.075 mmHg/day
        
        # Poor diet increases BP (high sodium)
        if diet < 0.5:
            bp_change += (0.5 - diet) * 0.1  # Up to +0.05 mmHg/day
        
        # Stress increases BP
        bp_change += stress * 0.08
        
        # Smoking increases BP
        if smoking > 0:
            bp_change += 0.05
        
        # Age-related increase
        bp_change += 0.01
        
        # Apply BP changes
        self.systolic_bp += bp_change
        self.diastolic_bp += bp_change * 0.6  # Diastolic changes less
        
        # Physiological bounds
        self.systolic_bp = max(90, min(self.systolic_bp, 200))
        self.diastolic_bp = max(60, min(self.diastolic_bp, 120))
        
        # === CHOLESTEROL EVOLUTION ===
        ldl_change = 0.0
        hdl_change = 0.0
        
        # Poor diet increases LDL
        if diet < 0.5:
            ldl_change += (0.5 - diet) * 0.2  # Up to +0.1 mg/dL/day
        
        # Exercise decreases LDL, increases HDL
        if exercise > 0.5:
            ldl_change -= (exercise - 0.5) * 0.15
            hdl_change += (exercise - 0.5) * 0.1
        
        # High glucose increases triglycerides and LDL
        if glucose > 1.0:
            ldl_change += (glucose - 1.0) * 0.1
            self.triglycerides += (glucose - 1.0) * 0.3
        
        # Apply cholesterol changes
        self.ldl += ldl_change
        self.hdl += hdl_change
        self.total_cholesterol = self.ldl + self.hdl + (self.triglycerides / 5)
        
        # Bounds
        self.ldl = max(50, min(self.ldl, 300))
        self.hdl = max(30, min(self.hdl, 100))
        self.triglycerides = max(50, min(self.triglycerides, 500))
        
        # === VESSEL HEALTH EVOLUTION ===
        # Age-related decline
        self.vessel_elasticity *= 0.99995
        
        # Smoking damages vessels
        if smoking > 0:
            self.vessel_elasticity *= 0.9998
            self.endothelial_function *= 0.9998
        
        # High BP damages vessels
        if self.systolic_bp > 140:
            self.vessel_elasticity *= 0.9997
        
        # Exercise improves vessel health
        if exercise > 0.6:
            self.endothelial_function = min(1.0, self.endothelial_function * 1.0001)
        
        # === ATHEROSCLEROSIS PROGRESSION ===
        athero_increase = 0.0
        
        # High LDL promotes atherosclerosis
        if self.ldl > 130:
            athero_increase += (self.ldl - 130) * 0.00001
        
        # Low HDL promotes atherosclerosis
        if self.hdl < 40:
            athero_increase += (40 - self.hdl) * 0.00001
        
        # High glucose promotes atherosclerosis
        if glucose > 1.0:
            athero_increase += (glucose - 1.0) * 0.0001
        
        # Smoking accelerates atherosclerosis
        if smoking > 0:
            athero_increase += 0.0002
        
        # Exercise slows atherosclerosis
        if exercise > 0.6:
            athero_increase *= 0.7
        
        self.atherosclerosis_level += athero_increase
        self.atherosclerosis_level = min(1.0, self.atherosclerosis_level)
        
        # Atherosclerosis increases BP
        self.systolic_bp += self.atherosclerosis_level * 0.5
        
        # Record state
        self.state.record({
            'systolic_bp': self.systolic_bp,
            'diastolic_bp': self.diastolic_bp,
            'ldl': self.ldl,
            'hdl': self.hdl,
            'total_cholesterol': self.total_cholesterol,
            'triglycerides': self.triglycerides,
            'atherosclerosis': self.atherosclerosis_level,
            'vessel_elasticity': self.vessel_elasticity,
            'endothelial_function': self.endothelial_function,
            'lifestyle_impact': {
                'exercise': exercise,
                'diet': diet,
                'smoking': smoking
            }
        })
        
        return {
            'blood_pressure': {
                'systolic': self.systolic_bp,
                'diastolic': self.diastolic_bp
            },
            'vascular_stress': self.atherosclerosis_level,
            'cholesterol_stress': max(0, (self.ldl - 100) / 100)
        }
    
    def predict_disease(self) -> List[Dict]:
        """Predict cardiovascular disease risk"""
        predictions = []
        
        # Framingham Risk Score (simplified)
        cvd_risk = self._calculate_framingham_risk()
        
        predictions.append({
            'disease': 'Cardiovascular Disease',
            'probability': cvd_risk,
            'time_to_onset_years': 10.0,  # Framingham is 10-year risk
            'confidence': 0.80,
            'risk_factors': self._get_cvd_risk_factors()
        })
        
        # Hypertension prediction
        if self.systolic_bp >= 140 or self.diastolic_bp >= 90:
            hypertension_risk = 0.95
        elif self.systolic_bp >= 130:
            hypertension_risk = 0.60
        else:
            hypertension_risk = min(0.3, (self.systolic_bp - 110) / 100)
        
        predictions.append({
            'disease': 'Hypertension',
            'probability': hypertension_risk,
            'time_to_onset_years': 2.0 if hypertension_risk > 0.5 else 5.0,
            'confidence': 0.85,
            'risk_factors': [f"BP {int(self.systolic_bp)}/{int(self.diastolic_bp)}"]
        })
        
        return predictions
    
    def _calculate_framingham_risk(self) -> float:
        """Simplified Framingham Risk Score"""
        # This is a simplified version - real Framingham is more complex
        risk = 0.0
        
        # Age
        if self.age >= 60:
            risk += 0.3
        elif self.age >= 50:
            risk += 0.2
        elif self.age >= 40:
            risk += 0.1
        
        # Sex (males higher risk)
        if self.sex == 'M':
            risk += 0.1
        
        # Cholesterol
        if self.total_cholesterol > 240:
            risk += 0.2
        elif self.total_cholesterol > 200:
            risk += 0.1
        
        # HDL (protective)
        if self.hdl < 40:
            risk += 0.1
        elif self.hdl > 60:
            risk -= 0.05
        
        # Blood pressure
        if self.systolic_bp >= 160:
            risk += 0.2
        elif self.systolic_bp >= 140:
            risk += 0.1
        
        # Smoking
        if self.smoking:
            risk += 0.2
        
        return min(risk, 1.0)
    
    def _get_cvd_risk_factors(self) -> List[str]:
        """Identify CVD risk factors"""
        factors = []
        
        if self.systolic_bp >= 140:
            factors.append(f"High BP ({int(self.systolic_bp)}/{int(self.diastolic_bp)})")
        if self.ldl > 130:
            factors.append(f"High LDL ({int(self.ldl)} mg/dL)")
        if self.hdl < 40:
            factors.append(f"Low HDL ({int(self.hdl)} mg/dL)")
        if self.smoking:
            factors.append("Current smoker")
        if self.age >= 55:
            factors.append(f"Age {self.age}")
        
        return factors


# Continue with other agents...
class HepaticAgent:
    """Manages liver function and metabolism"""
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        self.alt = patient_data.get('alt', 25)
        self.ast = patient_data.get('ast', 25)
        self.fat_accumulation = 0.0
        self.liver_function = 1.0  # 0-1 scale
        self.age = patient_data.get('age', 40)
        self.last_perception = {}
        
    def perceive(self, signals: Dict):
        """Receive metabolic signals"""
        self.last_perception = signals
        
        if 'glucose_level' in signals:
            if signals['glucose_level'] > 1.2:
                self.fat_accumulation += 0.01
        
        if 'metabolic_stress' in signals:
            self.alt += signals['metabolic_stress'] * 0.5
            self.ast += signals['metabolic_stress'] * 0.5
        
        if 'alcohol_consumption' in signals:
            self.alt += signals['alcohol_consumption'] * 0.5
            self.ast += signals['alcohol_consumption'] * 0.5
        
        if 'diet_quality' in signals:
            if signals['diet_quality'] < 0.5:
                self.fat_accumulation += 0.005
        
        if 'exercise_level' in signals:
            if signals['exercise_level'] > 0.5:
                self.fat_accumulation -= 0.005
        
    def act(self) -> Dict:
        """Update liver state based on lifestyle and metabolic signals"""
        # Get lifestyle factors
        diet = self.last_perception.get('diet_quality', 0.5)
        exercise = self.last_perception.get('exercise_level', 0.5)
        alcohol = self.last_perception.get('alcohol_consumption', 0.0)
        
        # Liver enzyme evolution
        alt_change = 0.0
        ast_change = 0.0
        
        # Poor diet increases liver enzymes
        if diet < 0.5:
            alt_change += (0.5 - diet) * 0.1
            ast_change += (0.5 - diet) * 0.1
        
        # Exercise decreases liver enzymes
        if exercise > 0.5:
            alt_change -= (exercise - 0.5) * 0.05
            ast_change -= (exercise - 0.5) * 0.05
        
        # Alcohol consumption increases liver enzymes
        alt_change += alcohol * 0.1
        ast_change += alcohol * 0.1
        
        # Apply liver enzyme changes
        self.alt += alt_change
        self.ast += ast_change
        
        # Physiological bounds
        self.alt = max(10, min(self.alt, 100))
        self.ast = max(10, min(self.ast, 100))
        
        # Liver fat accumulation evolution
        fat_change = 0.0
        
        # Poor diet increases liver fat
        if diet < 0.5:
            fat_change += (0.5 - diet) * 0.005
        
        # Exercise decreases liver fat
        if exercise > 0.5:
            fat_change -= (exercise - 0.5) * 0.005
        
        # Apply liver fat changes
        self.fat_accumulation += fat_change
        self.fat_accumulation = max(0.0, min(self.fat_accumulation, 1.0))
        
        # Liver function evolution
        liver_function_change = 0.0
        
        # Liver fat accumulation decreases liver function
        if self.fat_accumulation > 0.5:
            liver_function_change -= (self.fat_accumulation - 0.5) * 0.01
        
        # Apply liver function changes
        self.liver_function += liver_function_change
        self.liver_function = max(0.0, min(self.liver_function, 1.0))
        
        # Record state
        self.state.record({
            'alt': self.alt,
            'ast': self.ast,
            'fat_accumulation': self.fat_accumulation,
            'liver_function': self.liver_function,
            'lifestyle_impact': {
                'diet': diet,
                'exercise': exercise,
                'alcohol': alcohol
            }
        })
        
        return {
            'liver_enzymes': {
                'alt': self.alt,
                'ast': self.ast
            },
            'liver_fat': self.fat_accumulation,
            'liver_function': self.liver_function
        }
        return {'liver_fat': self.fat_accumulation}
    
    def predict_disease(self) -> Dict:
        risk = min((self.alt - 25) / 100, 1.0) if self.alt > 25 else 0.0
        return {
            'disease': 'Fatty Liver Disease',
            'probability': risk,
            'time_to_onset_years': 5.0,
            'confidence': 0.70
        }


class RenalAgent:
    """Kidney function agent"""
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        self.creatinine = patient_data.get('creatinine', 1.0)
        self.egfr = patient_data.get('egfr', self._calculate_egfr(patient_data))
        
    def _calculate_egfr(self, data: Dict) -> float:
        """Calculate eGFR using CKD-EPI equation"""
        creatinine = data.get('creatinine', 1.0)
        age = data.get('age', 40)
        sex = data.get('sex', 'M')
        
        # Simplified CKD-EPI
        if sex == 'F':
            egfr = 144 * (creatinine / 0.7) ** -0.329 * (0.993 ** age)
        else:
            egfr = 141 * (creatinine / 0.9) ** -0.411 * (0.993 ** age)
        
        return max(egfr, 15)
    
    def perceive(self, signals: Dict):
        pass
    
    def act(self) -> Dict:
        self.state.record({'egfr': self.egfr})
        return {'kidney_function': self.egfr / 100}
    
    def predict_disease(self) -> Dict:
        if self.egfr < 60:
            risk = 0.8
        elif self.egfr < 90:
            risk = 0.3
        else:
            risk = 0.1
        
        return {
            'disease': 'Chronic Kidney Disease',
            'probability': risk,
            'time_to_onset_years': 5.0,
            'confidence': 0.75
        }


class ImmuneAgent:
    """Immune system and inflammation agent"""
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        self.crp = patient_data.get('crp', 1.0)
        self.wbc = patient_data.get('wbc_count', 7.0)
        
    def perceive(self, signals: Dict):
        pass
    
    def act(self) -> Dict:
        self.state.record({'crp': self.crp})
        return {'inflammation_level': self.crp / 10}
    
    def predict_disease(self) -> Dict:
        risk = min(self.crp / 10, 0.5)
        return {
            'disease': 'Chronic Inflammation',
            'probability': risk,
            'time_to_onset_years': 3.0,
            'confidence': 0.65
        }


class NeuralAgent:
    """Brain and nervous system agent"""
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        self.cognitive_score = 1.0
        self.age = patient_data.get('age', 40)
        
    def perceive(self, signals: Dict):
        pass
    
    def act(self) -> Dict:
        # Age-related cognitive decline
        if self.age > 60:
            self.cognitive_score *= 0.9998
        
        self.state.record({'cognitive_score': self.cognitive_score})
        return {'cognitive_function': self.cognitive_score}
    
    def predict_disease(self) -> Dict:
        if self.age < 60:
            risk = 0.05
        elif self.age < 70:
            risk = 0.15
        else:
            risk = 0.30
        
        return {
            'disease': 'Cognitive Decline',
            'probability': risk,
            'time_to_onset_years': max(70 - self.age, 1),
            'confidence': 0.60
        }


class EndocrineAgent:
    """Hormone and endocrine system agent"""
    
    def __init__(self, patient_data: Dict):
        self.state = AgentState()
        self.tsh = patient_data.get('tsh', 2.0)
        
    def perceive(self, signals: Dict):
        pass
    
    def act(self) -> Dict:
        self.state.record({'tsh': self.tsh})
        return {'thyroid_function': 1.0 if 0.5 < self.tsh < 4.5 else 0.5}
    
    def predict_disease(self) -> Dict:
        if self.tsh < 0.5 or self.tsh > 4.5:
            risk = 0.7
        else:
            risk = 0.1
        
        return {
            'disease': 'Thyroid Disorder',
            'probability': risk,
            'time_to_onset_years': 3.0,
            'confidence': 0.70
        }
