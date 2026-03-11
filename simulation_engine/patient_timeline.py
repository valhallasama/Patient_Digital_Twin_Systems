"""
Patient Timeline Engine - Temporal State Modeling
Models patient health state evolution over time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthState:
    """Patient health state at a specific time point"""
    timestamp: datetime
    
    # Physical measurements
    weight_kg: float
    bmi: float
    waist_circumference_cm: float
    
    # Vital signs
    systolic_bp: float
    diastolic_bp: float
    heart_rate: int
    
    # Metabolic markers
    glucose_mmol_l: float
    hba1c_percent: float
    insulin_resistance_index: float
    beta_cell_function_percent: float
    
    # Lipid profile
    total_cholesterol_mmol_l: float
    ldl_cholesterol_mmol_l: float
    hdl_cholesterol_mmol_l: float
    triglycerides_mmol_l: float
    
    # Organ function
    egfr: float
    liver_function_index: float
    
    # Disease states
    has_diabetes: bool = False
    has_cvd: bool = False
    has_hypertension: bool = False
    has_ckd: bool = False
    
    # Lifestyle factors
    exercise_hours_per_week: float = 0.0
    smoking_status: str = "never"
    alcohol_units_per_week: float = 0.0
    sleep_hours_per_night: float = 7.0
    stress_level: float = 0.5  # 0-1 scale
    
    # Medications
    on_metformin: bool = False
    on_statin: bool = False
    on_antihypertensive: bool = False
    
    # Computed risk scores
    cvd_risk_10yr: float = 0.0
    diabetes_risk: float = 0.0
    mortality_risk_5yr: float = 0.0


@dataclass
class PatientTimeline:
    """Complete patient timeline with state history"""
    patient_id: str
    birth_date: datetime
    gender: str
    ethnicity: str
    
    # Timeline of health states
    states: List[HealthState] = field(default_factory=list)
    
    # Major events
    events: List[Dict] = field(default_factory=list)
    
    # Interventions
    interventions: List[Dict] = field(default_factory=list)
    
    def add_state(self, state: HealthState):
        """Add health state to timeline"""
        self.states.append(state)
        self.states.sort(key=lambda s: s.timestamp)
    
    def add_event(self, timestamp: datetime, event_type: str, description: str):
        """Record major health event"""
        self.events.append({
            'timestamp': timestamp,
            'type': event_type,
            'description': description
        })
    
    def add_intervention(self, timestamp: datetime, intervention_type: str, details: Dict):
        """Record intervention"""
        self.interventions.append({
            'timestamp': timestamp,
            'type': intervention_type,
            'details': details
        })
    
    def get_state_at_time(self, timestamp: datetime) -> Optional[HealthState]:
        """Get health state at specific time (or closest before)"""
        valid_states = [s for s in self.states if s.timestamp <= timestamp]
        return valid_states[-1] if valid_states else None
    
    def get_trajectory(self, parameter: str) -> pd.DataFrame:
        """Get trajectory of specific parameter over time"""
        data = []
        for state in self.states:
            if hasattr(state, parameter):
                data.append({
                    'timestamp': state.timestamp,
                    'value': getattr(state, parameter)
                })
        return pd.DataFrame(data)


class PatientTimelineEngine:
    """
    Simulates patient health state evolution over time
    Uses mechanistic models, not just static risk scores
    """
    
    def __init__(self):
        self.timelines: Dict[str, PatientTimeline] = {}
    
    def create_timeline(self, patient_id: str, initial_state: HealthState, 
                       birth_date: datetime, gender: str, ethnicity: str) -> PatientTimeline:
        """Create new patient timeline"""
        timeline = PatientTimeline(
            patient_id=patient_id,
            birth_date=birth_date,
            gender=gender,
            ethnicity=ethnicity
        )
        timeline.add_state(initial_state)
        self.timelines[patient_id] = timeline
        return timeline
    
    def simulate_forward(self, patient_id: str, years: int = 10, 
                        timestep_months: int = 6) -> PatientTimeline:
        """
        Simulate patient health state forward in time
        Uses state transition models
        """
        timeline = self.timelines[patient_id]
        current_state = timeline.states[-1]
        current_time = current_state.timestamp
        
        num_steps = int(years * 12 / timestep_months)
        
        for step in range(num_steps):
            # Advance time
            current_time += timedelta(days=30 * timestep_months)
            
            # Simulate next state
            next_state = self._transition_state(current_state, timestep_months, timeline)
            next_state.timestamp = current_time
            
            # Check for disease onset
            self._check_disease_onset(current_state, next_state, timeline, current_time)
            
            # Add to timeline
            timeline.add_state(next_state)
            current_state = next_state
        
        return timeline
    
    def _transition_state(self, current: HealthState, months: int, 
                         timeline: PatientTimeline) -> HealthState:
        """
        State transition function: state(t+1) = f(state(t), lifestyle, environment)
        This is the core mechanistic model
        """
        
        # Calculate age
        age = (current.timestamp - timeline.birth_date).days / 365.25
        
        # --- Weight dynamics ---
        # Caloric balance affects weight
        exercise_effect = -0.05 * current.exercise_hours_per_week
        age_effect = 0.02 * (age - 30) / 10 if age > 30 else 0
        stress_effect = 0.03 * current.stress_level
        
        weight_change_rate = age_effect + stress_effect + exercise_effect
        new_weight = current.weight_kg * (1 + weight_change_rate * months / 12)
        new_weight = np.clip(new_weight, 40, 200)
        
        # BMI
        height_m = np.sqrt(current.weight_kg / current.bmi)  # Reverse calculate height
        new_bmi = new_weight / (height_m ** 2)
        
        # --- Glucose homeostasis ---
        # Beta cell function declines with age and insulin resistance
        ir_progression = 0.01 * (new_bmi - 25) / 5 if new_bmi > 25 else 0
        ir_progression += 0.005 * age / 50
        ir_progression -= 0.01 * current.exercise_hours_per_week / 5
        
        new_ir_index = current.insulin_resistance_index * (1 + ir_progression * months / 12)
        new_ir_index = np.clip(new_ir_index, 0.5, 5.0)
        
        # Beta cell function
        beta_decline_rate = 0.01 * new_ir_index  # Insulin resistance damages beta cells
        beta_decline_rate += 0.005 if age > 50 else 0
        
        new_beta_function = current.beta_cell_function_percent * (1 - beta_decline_rate * months / 12)
        new_beta_function = np.clip(new_beta_function, 10, 100)
        
        # Glucose from insulin resistance and beta cell function
        glucose_base = 5.0
        glucose_ir_effect = (new_ir_index - 1.0) * 1.5
        glucose_beta_effect = (100 - new_beta_function) / 100 * 3.0
        
        new_glucose = glucose_base + glucose_ir_effect + glucose_beta_effect
        new_glucose = np.clip(new_glucose, 3.5, 15.0)
        
        # HbA1c from average glucose
        new_hba1c = (new_glucose - 3.5) / 1.8 + 4.0
        new_hba1c = np.clip(new_hba1c, 4.0, 12.0)
        
        # --- Blood pressure dynamics ---
        # Affected by weight, age, stress, exercise
        bp_age_effect = 0.5 * (age - 30) / 10 if age > 30 else 0
        bp_weight_effect = 1.0 * (new_bmi - 25) / 5 if new_bmi > 25 else 0
        bp_stress_effect = 5.0 * current.stress_level
        bp_exercise_effect = -2.0 * current.exercise_hours_per_week / 5
        bp_medication_effect = -15.0 if current.on_antihypertensive else 0
        
        new_systolic = current.systolic_bp + (bp_age_effect + bp_weight_effect + 
                                              bp_stress_effect + bp_exercise_effect + 
                                              bp_medication_effect) * months / 12
        new_systolic = np.clip(new_systolic, 90, 200)
        
        new_diastolic = new_systolic * 0.65  # Approximate relationship
        new_diastolic = np.clip(new_diastolic, 60, 120)
        
        # --- Lipid dynamics ---
        # Affected by diet, exercise, weight, medications
        ldl_weight_effect = 0.1 * (new_bmi - 25) / 5 if new_bmi > 25 else 0
        ldl_exercise_effect = -0.05 * current.exercise_hours_per_week / 5
        ldl_statin_effect = -1.5 if current.on_statin else 0
        
        new_ldl = current.ldl_cholesterol_mmol_l + (ldl_weight_effect + ldl_exercise_effect + 
                                                     ldl_statin_effect) * months / 12
        new_ldl = np.clip(new_ldl, 1.0, 8.0)
        
        hdl_exercise_effect = 0.05 * current.exercise_hours_per_week / 5
        new_hdl = current.hdl_cholesterol_mmol_l + hdl_exercise_effect * months / 12
        new_hdl = np.clip(new_hdl, 0.5, 3.0)
        
        new_total_chol = new_ldl + new_hdl + 0.5
        new_triglycerides = current.triglycerides_mmol_l * (1 + 0.01 * (new_bmi - 25) / 5)
        new_triglycerides = np.clip(new_triglycerides, 0.5, 5.0)
        
        # --- Kidney function ---
        # Declines with age, diabetes, hypertension
        egfr_age_decline = -0.5 * (age - 40) / 10 if age > 40 else 0
        egfr_diabetes_effect = -1.0 if current.has_diabetes else 0
        egfr_htn_effect = -0.5 if current.has_hypertension else 0
        
        new_egfr = current.egfr + (egfr_age_decline + egfr_diabetes_effect + 
                                   egfr_htn_effect) * months / 12
        new_egfr = np.clip(new_egfr, 15, 120)
        
        # Create new state
        new_state = HealthState(
            timestamp=current.timestamp,  # Will be updated by caller
            weight_kg=new_weight,
            bmi=new_bmi,
            waist_circumference_cm=current.waist_circumference_cm * (new_weight / current.weight_kg) ** 0.5,
            systolic_bp=new_systolic,
            diastolic_bp=new_diastolic,
            heart_rate=current.heart_rate,
            glucose_mmol_l=new_glucose,
            hba1c_percent=new_hba1c,
            insulin_resistance_index=new_ir_index,
            beta_cell_function_percent=new_beta_function,
            total_cholesterol_mmol_l=new_total_chol,
            ldl_cholesterol_mmol_l=new_ldl,
            hdl_cholesterol_mmol_l=new_hdl,
            triglycerides_mmol_l=new_triglycerides,
            egfr=new_egfr,
            liver_function_index=current.liver_function_index,
            has_diabetes=current.has_diabetes,
            has_cvd=current.has_cvd,
            has_hypertension=current.has_hypertension,
            has_ckd=current.has_ckd,
            exercise_hours_per_week=current.exercise_hours_per_week,
            smoking_status=current.smoking_status,
            alcohol_units_per_week=current.alcohol_units_per_week,
            sleep_hours_per_night=current.sleep_hours_per_night,
            stress_level=current.stress_level,
            on_metformin=current.on_metformin,
            on_statin=current.on_statin,
            on_antihypertensive=current.on_antihypertensive
        )
        
        return new_state
    
    def _check_disease_onset(self, prev_state: HealthState, new_state: HealthState,
                            timeline: PatientTimeline, timestamp: datetime):
        """Check for disease onset based on state transitions"""
        
        # Diabetes onset
        if not prev_state.has_diabetes and new_state.hba1c_percent >= 6.5:
            new_state.has_diabetes = True
            timeline.add_event(timestamp, 'disease_onset', 'Type 2 Diabetes diagnosed')
            logger.info(f"Patient {timeline.patient_id}: Diabetes onset at {timestamp}")
        
        # Hypertension onset
        if not prev_state.has_hypertension and new_state.systolic_bp >= 140:
            new_state.has_hypertension = True
            timeline.add_event(timestamp, 'disease_onset', 'Hypertension diagnosed')
            logger.info(f"Patient {timeline.patient_id}: Hypertension onset at {timestamp}")
        
        # CKD onset
        if not prev_state.has_ckd and new_state.egfr < 60:
            new_state.has_ckd = True
            timeline.add_event(timestamp, 'disease_onset', 'Chronic Kidney Disease diagnosed')
            logger.info(f"Patient {timeline.patient_id}: CKD onset at {timestamp}")
    
    def apply_intervention(self, patient_id: str, intervention_type: str, 
                          parameters: Dict, start_time: datetime):
        """Apply intervention and modify future states"""
        timeline = self.timelines[patient_id]
        timeline.add_intervention(start_time, intervention_type, parameters)
        
        # Modify current state based on intervention
        current_state = timeline.get_state_at_time(start_time)
        if current_state:
            if intervention_type == 'exercise_program':
                current_state.exercise_hours_per_week = parameters.get('hours_per_week', 5.0)
            elif intervention_type == 'medication_metformin':
                current_state.on_metformin = True
            elif intervention_type == 'medication_statin':
                current_state.on_statin = True
            elif intervention_type == 'medication_antihypertensive':
                current_state.on_antihypertensive = True
            elif intervention_type == 'lifestyle_modification':
                current_state.exercise_hours_per_week = parameters.get('exercise', 3.0)
                current_state.stress_level = parameters.get('stress_level', 0.3)
        
        logger.info(f"Applied intervention: {intervention_type} to patient {patient_id}")
    
    def compare_scenarios(self, patient_id: str, interventions: List[Dict], 
                         years: int = 10) -> Dict[str, PatientTimeline]:
        """
        Compare multiple intervention scenarios
        Returns dictionary of timelines for each scenario
        """
        scenarios = {}
        
        # Baseline (no intervention)
        baseline_timeline = self.timelines[patient_id]
        scenarios['baseline'] = self.simulate_forward(patient_id, years)
        
        # Each intervention scenario
        for i, intervention in enumerate(interventions):
            # Create copy of timeline
            scenario_id = f"{patient_id}_scenario_{i}"
            # ... implementation for scenario comparison
        
        return scenarios


# Example usage
if __name__ == "__main__":
    # Create patient timeline engine
    engine = PatientTimelineEngine()
    
    # Initial health state
    initial_state = HealthState(
        timestamp=datetime(2024, 1, 1),
        weight_kg=85,
        bmi=28,
        waist_circumference_cm=95,
        systolic_bp=135,
        diastolic_bp=85,
        heart_rate=72,
        glucose_mmol_l=5.8,
        hba1c_percent=5.7,
        insulin_resistance_index=1.5,
        beta_cell_function_percent=85,
        total_cholesterol_mmol_l=5.2,
        ldl_cholesterol_mmol_l=3.5,
        hdl_cholesterol_mmol_l=1.2,
        triglycerides_mmol_l=1.8,
        egfr=95,
        liver_function_index=1.0,
        exercise_hours_per_week=2.0,
        smoking_status='never',
        sleep_hours_per_night=6.5,
        stress_level=0.6
    )
    
    # Create timeline
    timeline = engine.create_timeline(
        patient_id='P001',
        initial_state=initial_state,
        birth_date=datetime(1972, 5, 15),
        gender='male',
        ethnicity='caucasian'
    )
    
    # Simulate 10 years forward
    print("\n" + "="*80)
    print("SIMULATING PATIENT HEALTH TRAJECTORY - 10 YEARS")
    print("="*80)
    
    timeline = engine.simulate_forward('P001', years=10, timestep_months=6)
    
    # Display trajectory
    print(f"\nPatient P001 - 10-year health trajectory:")
    print(f"States recorded: {len(timeline.states)}")
    print(f"Events: {len(timeline.events)}")
    
    # Show key parameters over time
    for i, state in enumerate(timeline.states[::4]):  # Every 2 years
        years_elapsed = i * 2
        print(f"\nYear {years_elapsed}:")
        print(f"  BMI: {state.bmi:.1f}")
        print(f"  HbA1c: {state.hba1c_percent:.1f}%")
        print(f"  BP: {state.systolic_bp:.0f}/{state.diastolic_bp:.0f}")
        print(f"  LDL: {state.ldl_cholesterol_mmol_l:.1f} mmol/L")
        print(f"  eGFR: {state.egfr:.0f}")
        print(f"  Diabetes: {state.has_diabetes}")
        print(f"  Hypertension: {state.has_hypertension}")
    
    # Show events
    if timeline.events:
        print("\n" + "="*80)
        print("MAJOR HEALTH EVENTS")
        print("="*80)
        for event in timeline.events:
            print(f"{event['timestamp'].year}: {event['description']}")
