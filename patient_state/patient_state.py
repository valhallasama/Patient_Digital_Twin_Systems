#!/usr/bin/env python3
"""
Unified Patient State Model
Core digital twin state representation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class Demographics:
    """Patient demographic information"""
    age: int
    sex: str
    ethnicity: Optional[str] = None
    genetics: Optional[Dict] = None
    family_history: Dict = field(default_factory=dict)


@dataclass
class Physiology:
    """Core physiological parameters"""
    # Metabolic
    bmi: float = 25.0
    weight: float = 70.0
    height: float = 170.0
    glucose: float = 100.0
    hba1c: float = 5.5
    insulin: float = 10.0
    
    # Cardiovascular
    blood_pressure: Dict = field(default_factory=lambda: {'systolic': 120, 'diastolic': 80})
    heart_rate: int = 70
    
    # Lipids
    total_cholesterol: float = 180.0
    ldl_cholesterol: float = 100.0
    hdl_cholesterol: float = 50.0
    triglycerides: float = 150.0
    
    # Liver
    alt: float = 30.0
    ast: float = 30.0
    
    # Kidney
    creatinine: float = 1.0
    egfr: float = 90.0


@dataclass
class OrganHealth:
    """Organ-specific health metrics"""
    # Cardiovascular
    heart_risk: float = 0.0
    vessel_elasticity: float = 1.0
    atherosclerosis_level: float = 0.0
    
    # Hepatic
    liver_fat: float = 0.0
    liver_function: float = 1.0
    
    # Renal
    kidney_function: float = 1.0
    kidney_damage: float = 0.0
    
    # Metabolic
    insulin_sensitivity: float = 1.0
    beta_cell_function: float = 1.0
    
    # Immune
    inflammation_level: float = 0.0
    immune_function: float = 1.0


@dataclass
class Lifestyle:
    """Lifestyle and behavioral factors"""
    exercise_level: str = 'moderate'  # sedentary, light, moderate, vigorous
    diet_quality: str = 'fair'  # poor, fair, good, excellent
    sleep_duration: float = 7.0
    stress_level: str = 'moderate'  # low, moderate, high
    smoking_status: str = 'never'  # never, former, current
    alcohol_consumption: str = 'none'  # none, light, moderate, heavy
    
    # Numeric conversions (0-1 scale)
    exercise_numeric: float = 0.5
    diet_numeric: float = 0.5
    stress_numeric: float = 0.5
    smoking_numeric: float = 0.0
    alcohol_numeric: float = 0.0


@dataclass
class MedicalHistory:
    """Medical history and conditions"""
    diagnoses: List[str] = field(default_factory=list)
    medications: List[Dict] = field(default_factory=list)
    surgeries: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    immunizations: List[str] = field(default_factory=list)


class PatientState:
    """
    Unified digital twin state representation
    This is the core data model for the patient digital twin
    """
    
    def __init__(self, patient_data: Dict):
        """
        Initialize patient state from raw data
        
        Args:
            patient_data: Raw patient data dictionary
        """
        self.patient_id = patient_data.get('patient_id', 'UNKNOWN')
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # Initialize all components
        self.demographics = self._init_demographics(patient_data)
        self.physiology = self._init_physiology(patient_data)
        self.organ_health = self._init_organ_health(patient_data)
        self.lifestyle = self._init_lifestyle(patient_data)
        self.medical_history = self._init_medical_history(patient_data)
        
        # State history for temporal tracking
        self.history: List[Dict] = []
        
        # Store raw data for reference
        self.raw_data = patient_data
    
    def _init_demographics(self, data: Dict) -> Demographics:
        """Initialize demographics from data"""
        return Demographics(
            age=data.get('age', 40),
            sex=data.get('sex', 'M'),
            ethnicity=data.get('ethnicity'),
            genetics=data.get('genetics'),
            family_history=data.get('family_history', {})
        )
    
    def _init_physiology(self, data: Dict) -> Physiology:
        """Initialize physiology from data"""
        return Physiology(
            bmi=data.get('bmi', 25.0),
            weight=data.get('weight', 70.0),
            height=data.get('height', 170.0),
            glucose=data.get('fasting_glucose', 100.0),
            hba1c=data.get('hba1c', 5.5),
            insulin=data.get('insulin', 10.0),
            blood_pressure=data.get('blood_pressure', {'systolic': 120, 'diastolic': 80}),
            heart_rate=data.get('heart_rate', 70),
            total_cholesterol=data.get('total_cholesterol', 180.0),
            ldl_cholesterol=data.get('ldl_cholesterol', 100.0),
            hdl_cholesterol=data.get('hdl_cholesterol', 50.0),
            triglycerides=data.get('triglycerides', 150.0),
            alt=data.get('alt', 30.0),
            ast=data.get('ast', 30.0),
            creatinine=data.get('creatinine', 1.0),
            egfr=data.get('egfr', 90.0)
        )
    
    def _init_organ_health(self, data: Dict) -> OrganHealth:
        """Initialize organ health metrics"""
        return OrganHealth(
            heart_risk=data.get('heart_risk', 0.0),
            vessel_elasticity=data.get('vessel_elasticity', 1.0),
            atherosclerosis_level=data.get('atherosclerosis', 0.0),
            liver_fat=data.get('liver_fat', 0.0),
            liver_function=data.get('liver_function', 1.0),
            kidney_function=data.get('kidney_function', 1.0),
            kidney_damage=data.get('kidney_damage', 0.0),
            insulin_sensitivity=data.get('insulin_sensitivity', 1.0),
            beta_cell_function=data.get('beta_cell_function', 1.0),
            inflammation_level=data.get('inflammation', 0.0),
            immune_function=data.get('immune_function', 1.0)
        )
    
    def _init_lifestyle(self, data: Dict) -> Lifestyle:
        """Initialize lifestyle factors"""
        lifestyle_data = data.get('lifestyle', {})
        
        # Get categorical values
        exercise = lifestyle_data.get('physical_activity', 'moderate')
        diet = lifestyle_data.get('diet_quality', 'fair')
        stress = lifestyle_data.get('stress_level', 'moderate')
        smoking = lifestyle_data.get('smoking_status', 'never')
        alcohol = lifestyle_data.get('alcohol_consumption', 'none')
        
        # Convert to numeric
        lifestyle = Lifestyle(
            exercise_level=exercise,
            diet_quality=diet,
            sleep_duration=lifestyle_data.get('sleep_duration', 7.0),
            stress_level=stress,
            smoking_status=smoking,
            alcohol_consumption=alcohol
        )
        
        # Set numeric values
        lifestyle.exercise_numeric = self._lifestyle_to_numeric(exercise, 'exercise')
        lifestyle.diet_numeric = self._lifestyle_to_numeric(diet, 'quality')
        lifestyle.stress_numeric = self._lifestyle_to_numeric(stress, 'stress')
        lifestyle.smoking_numeric = 1.0 if smoking == 'current' else 0.0
        lifestyle.alcohol_numeric = self._lifestyle_to_numeric(alcohol, 'alcohol')
        
        return lifestyle
    
    def _lifestyle_to_numeric(self, value: str, category: str) -> float:
        """Convert lifestyle categories to numeric 0-1"""
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
    
    def _init_medical_history(self, data: Dict) -> MedicalHistory:
        """Initialize medical history"""
        history_data = data.get('medical_history', {})
        
        return MedicalHistory(
            diagnoses=history_data.get('diagnoses', []),
            medications=history_data.get('medications', []),
            surgeries=history_data.get('surgeries', []),
            allergies=history_data.get('allergies', []),
            immunizations=history_data.get('immunizations', [])
        )
    
    def snapshot(self) -> Dict:
        """Create a snapshot of current state"""
        return {
            'timestamp': datetime.now().isoformat(),
            'demographics': {
                'age': self.demographics.age,
                'sex': self.demographics.sex
            },
            'physiology': {
                'bmi': self.physiology.bmi,
                'glucose': self.physiology.glucose,
                'hba1c': self.physiology.hba1c,
                'blood_pressure': self.physiology.blood_pressure,
                'ldl': self.physiology.ldl_cholesterol,
                'hdl': self.physiology.hdl_cholesterol
            },
            'organ_health': {
                'heart_risk': self.organ_health.heart_risk,
                'liver_fat': self.organ_health.liver_fat,
                'kidney_function': self.organ_health.kidney_function,
                'insulin_sensitivity': self.organ_health.insulin_sensitivity
            },
            'lifestyle': {
                'exercise': self.lifestyle.exercise_level,
                'diet': self.lifestyle.diet_quality,
                'stress': self.lifestyle.stress_level,
                'smoking': self.lifestyle.smoking_status
            }
        }
    
    def record_snapshot(self):
        """Record current state to history"""
        self.history.append(self.snapshot())
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert entire state to dictionary"""
        return {
            'patient_id': self.patient_id,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'demographics': self.demographics.__dict__,
            'physiology': self.physiology.__dict__,
            'organ_health': self.organ_health.__dict__,
            'lifestyle': self.lifestyle.__dict__,
            'medical_history': self.medical_history.__dict__,
            'history_length': len(self.history)
        }
    
    def to_json(self) -> str:
        """Convert state to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def get_risk_summary(self) -> Dict:
        """Get summary of current risk factors"""
        risks = []
        
        # Metabolic risks
        if self.physiology.hba1c >= 6.5:
            risks.append('Diabetes (HbA1c ≥ 6.5%)')
        elif self.physiology.hba1c >= 5.7:
            risks.append('Prediabetes (HbA1c 5.7-6.4%)')
        
        if self.physiology.bmi >= 30:
            risks.append('Obesity (BMI ≥ 30)')
        elif self.physiology.bmi >= 25:
            risks.append('Overweight (BMI 25-30)')
        
        # Cardiovascular risks
        bp = self.physiology.blood_pressure
        if bp['systolic'] >= 140 or bp['diastolic'] >= 90:
            risks.append('Hypertension (BP ≥ 140/90)')
        
        if self.physiology.ldl_cholesterol >= 160:
            risks.append('High LDL (≥ 160 mg/dL)')
        
        # Lifestyle risks
        if self.lifestyle.smoking_status == 'current':
            risks.append('Current smoker')
        
        if self.lifestyle.exercise_level == 'sedentary':
            risks.append('Sedentary lifestyle')
        
        return {
            'risk_count': len(risks),
            'risk_factors': risks,
            'risk_level': 'high' if len(risks) >= 4 else 'moderate' if len(risks) >= 2 else 'low'
        }
