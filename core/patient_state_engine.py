"""
Centralized Patient State Engine
Continuous temporal tracking of patient health state
Stores time-series, updates from agents, enables forward/backward simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatientStateSnapshot:
    """Single point-in-time patient state"""
    timestamp: datetime
    patient_id: str
    
    # Demographics (static)
    age: float
    gender: str
    ethnicity: str
    
    # Physical measurements
    weight_kg: float
    height_cm: float
    bmi: float
    waist_cm: float
    
    # Vital signs
    systolic_bp: float
    diastolic_bp: float
    heart_rate: int
    temperature_c: float
    
    # Laboratory values
    glucose_mmol_l: float
    hba1c_percent: float
    total_chol_mmol_l: float
    ldl_mmol_l: float
    hdl_mmol_l: float
    triglycerides_mmol_l: float
    creatinine_umol_l: float
    egfr: float
    alt: float
    ast: float
    
    # Metabolic state
    insulin_resistance_index: float
    beta_cell_function_percent: float
    
    # Disease states
    has_diabetes: bool = False
    has_cvd: bool = False
    has_hypertension: bool = False
    has_ckd: bool = False
    
    # Lifestyle
    smoking_status: str = "never"
    exercise_hours_week: float = 0.0
    alcohol_units_week: float = 0.0
    sleep_hours_night: float = 7.0
    stress_level: float = 0.5
    diet_quality_score: float = 0.5
    
    # Medications
    medications: List[str] = field(default_factory=list)
    
    # Agent assessments (updated by agents)
    cardiology_risk_score: Optional[float] = None
    metabolic_risk_score: Optional[float] = None
    lifestyle_risk_score: Optional[float] = None
    
    # ML predictions (updated by models)
    ml_diabetes_risk: Optional[float] = None
    ml_cvd_risk: Optional[float] = None
    ml_mortality_risk: Optional[float] = None
    
    # Metadata
    data_source: str = "manual"  # manual, ehr, wearable, simulation
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class PatientStateEngine:
    """
    Centralized engine for managing patient state over time
    Enables continuous tracking, updates from multiple sources,
    and temporal queries
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/patient_states")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory state storage (patient_id -> list of states)
        self.patient_states: Dict[str, List[PatientStateSnapshot]] = {}
        
        # Active patients (currently being tracked)
        self.active_patients: set = set()
        
        # State update callbacks (for real-time updates)
        self.update_callbacks: List[callable] = []
        
        logger.info(f"✓ Patient State Engine initialized")
        logger.info(f"  Storage: {self.storage_path}")
    
    def register_patient(self, patient_id: str, initial_state: PatientStateSnapshot):
        """Register new patient and store initial state"""
        if patient_id in self.patient_states:
            logger.warning(f"Patient {patient_id} already registered")
            return
        
        self.patient_states[patient_id] = [initial_state]
        self.active_patients.add(patient_id)
        
        logger.info(f"✓ Registered patient: {patient_id}")
        logger.info(f"  Initial state: {initial_state.timestamp}")
        
        # Persist to disk
        self._save_patient_state(patient_id)
    
    def update_state(self, patient_id: str, new_state: PatientStateSnapshot,
                    source: str = "update"):
        """
        Update patient state with new snapshot
        This is the core method for temporal tracking
        """
        if patient_id not in self.patient_states:
            raise ValueError(f"Patient {patient_id} not registered")
        
        # Add new state
        self.patient_states[patient_id].append(new_state)
        
        # Sort by timestamp
        self.patient_states[patient_id].sort(key=lambda s: s.timestamp)
        
        logger.debug(f"Updated state for {patient_id} at {new_state.timestamp}")
        
        # Trigger callbacks
        for callback in self.update_callbacks:
            callback(patient_id, new_state)
        
        # Persist
        self._save_patient_state(patient_id)
    
    def get_current_state(self, patient_id: str) -> Optional[PatientStateSnapshot]:
        """Get most recent state for patient"""
        if patient_id not in self.patient_states:
            return None
        
        states = self.patient_states[patient_id]
        return states[-1] if states else None
    
    def get_state_at_time(self, patient_id: str, 
                         timestamp: datetime) -> Optional[PatientStateSnapshot]:
        """Get patient state at specific time (or closest before)"""
        if patient_id not in self.patient_states:
            return None
        
        states = self.patient_states[patient_id]
        valid_states = [s for s in states if s.timestamp <= timestamp]
        
        return valid_states[-1] if valid_states else None
    
    def get_state_history(self, patient_id: str,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[PatientStateSnapshot]:
        """Get patient state history within time range"""
        if patient_id not in self.patient_states:
            return []
        
        states = self.patient_states[patient_id]
        
        if start_time:
            states = [s for s in states if s.timestamp >= start_time]
        if end_time:
            states = [s for s in states if s.timestamp <= end_time]
        
        return states
    
    def get_parameter_trajectory(self, patient_id: str, 
                                parameter: str) -> pd.DataFrame:
        """
        Get trajectory of specific parameter over time
        Returns DataFrame with timestamp and value
        """
        states = self.patient_states.get(patient_id, [])
        
        data = []
        for state in states:
            if hasattr(state, parameter):
                data.append({
                    'timestamp': state.timestamp,
                    'value': getattr(state, parameter)
                })
        
        return pd.DataFrame(data)
    
    def get_all_parameters_trajectory(self, patient_id: str) -> pd.DataFrame:
        """Get all parameters as time-series DataFrame"""
        states = self.patient_states.get(patient_id, [])
        
        if not states:
            return pd.DataFrame()
        
        data = [state.to_dict() for state in states]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        return df
    
    def update_from_agent_assessment(self, patient_id: str, 
                                    agent_name: str, 
                                    assessment: Dict):
        """
        Update patient state based on agent assessment
        Creates new state snapshot with agent's risk scores
        """
        current_state = self.get_current_state(patient_id)
        if not current_state:
            raise ValueError(f"No state found for patient {patient_id}")
        
        # Create new state with agent assessment
        new_state = PatientStateSnapshot(**asdict(current_state))
        new_state.timestamp = datetime.now()
        new_state.data_source = f"agent_{agent_name}"
        
        # Update agent-specific risk scores
        if agent_name.lower() == 'cardiology':
            new_state.cardiology_risk_score = assessment.get('risk_score', 0.5)
        elif agent_name.lower() == 'metabolic':
            new_state.metabolic_risk_score = assessment.get('risk_score', 0.5)
        elif agent_name.lower() == 'lifestyle':
            new_state.lifestyle_risk_score = assessment.get('risk_score', 0.5)
        
        self.update_state(patient_id, new_state, source=f"agent_{agent_name}")
        
        logger.info(f"✓ Updated state from {agent_name} agent")
    
    def update_from_ml_prediction(self, patient_id: str, predictions: Dict):
        """
        Update patient state with ML model predictions
        """
        current_state = self.get_current_state(patient_id)
        if not current_state:
            raise ValueError(f"No state found for patient {patient_id}")
        
        # Create new state with ML predictions
        new_state = PatientStateSnapshot(**asdict(current_state))
        new_state.timestamp = datetime.now()
        new_state.data_source = "ml_prediction"
        
        new_state.ml_diabetes_risk = predictions.get('diabetes_risk')
        new_state.ml_cvd_risk = predictions.get('cvd_risk')
        new_state.ml_mortality_risk = predictions.get('mortality_risk')
        
        self.update_state(patient_id, new_state, source="ml_prediction")
        
        logger.info(f"✓ Updated state with ML predictions")
    
    def update_from_simulation(self, patient_id: str, 
                              simulated_state: PatientStateSnapshot):
        """
        Update patient state from disease progression simulation
        """
        simulated_state.data_source = "simulation"
        self.update_state(patient_id, simulated_state, source="simulation")
        
        logger.info(f"✓ Updated state from simulation")
    
    def simulate_forward(self, patient_id: str, days: int = 365) -> PatientStateSnapshot:
        """
        Simulate patient state forward in time
        Uses simple mechanistic model
        """
        current_state = self.get_current_state(patient_id)
        if not current_state:
            raise ValueError(f"No state found for patient {patient_id}")
        
        # Create future state
        future_state = PatientStateSnapshot(**asdict(current_state))
        future_state.timestamp = current_state.timestamp + timedelta(days=days)
        
        # Simple progression model
        years = days / 365.25
        
        # Age progression
        future_state.age += years
        
        # Weight/BMI progression (age + lifestyle effects)
        weight_change_rate = 0.01  # 1% per year baseline
        if future_state.exercise_hours_week < 2:
            weight_change_rate += 0.02
        if future_state.stress_level > 0.6:
            weight_change_rate += 0.01
        
        future_state.weight_kg *= (1 + weight_change_rate * years)
        future_state.bmi = future_state.weight_kg / ((future_state.height_cm / 100) ** 2)
        
        # Glucose/HbA1c progression
        if future_state.bmi > 30:
            future_state.hba1c_percent += 0.1 * years
            future_state.glucose_mmol_l += 0.2 * years
        
        # BP progression
        future_state.systolic_bp += 0.5 * years
        
        # Check disease onset
        if future_state.hba1c_percent >= 6.5:
            future_state.has_diabetes = True
        if future_state.systolic_bp >= 140:
            future_state.has_hypertension = True
        
        future_state.data_source = "simulation_forward"
        
        return future_state
    
    def simulate_backward(self, patient_id: str, days: int = 365) -> PatientStateSnapshot:
        """
        Simulate patient state backward in time (retrospective)
        Useful for understanding disease progression history
        """
        current_state = self.get_current_state(patient_id)
        if not current_state:
            raise ValueError(f"No state found for patient {patient_id}")
        
        # Create past state
        past_state = PatientStateSnapshot(**asdict(current_state))
        past_state.timestamp = current_state.timestamp - timedelta(days=days)
        
        years = days / 365.25
        
        # Reverse progression
        past_state.age -= years
        past_state.weight_kg /= (1 + 0.01 * years)
        past_state.bmi = past_state.weight_kg / ((past_state.height_cm / 100) ** 2)
        past_state.hba1c_percent = max(4.5, past_state.hba1c_percent - 0.1 * years)
        past_state.systolic_bp = max(100, past_state.systolic_bp - 0.5 * years)
        
        past_state.data_source = "simulation_backward"
        
        return past_state
    
    def detect_state_changes(self, patient_id: str, 
                           lookback_days: int = 90) -> List[Dict]:
        """
        Detect significant changes in patient state
        Returns list of detected changes
        """
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_states = self.get_state_history(patient_id, start_time=cutoff_time)
        
        if len(recent_states) < 2:
            return []
        
        changes = []
        first_state = recent_states[0]
        last_state = recent_states[-1]
        
        # Check for significant parameter changes
        if abs(last_state.bmi - first_state.bmi) > 2:
            changes.append({
                'parameter': 'bmi',
                'change': last_state.bmi - first_state.bmi,
                'significance': 'high' if abs(last_state.bmi - first_state.bmi) > 5 else 'medium'
            })
        
        if abs(last_state.hba1c_percent - first_state.hba1c_percent) > 0.5:
            changes.append({
                'parameter': 'hba1c',
                'change': last_state.hba1c_percent - first_state.hba1c_percent,
                'significance': 'high'
            })
        
        if abs(last_state.systolic_bp - first_state.systolic_bp) > 10:
            changes.append({
                'parameter': 'systolic_bp',
                'change': last_state.systolic_bp - first_state.systolic_bp,
                'significance': 'high' if abs(last_state.systolic_bp - first_state.systolic_bp) > 20 else 'medium'
            })
        
        # Check for disease onset
        if last_state.has_diabetes and not first_state.has_diabetes:
            changes.append({
                'parameter': 'disease_onset',
                'change': 'diabetes',
                'significance': 'critical'
            })
        
        return changes
    
    def register_update_callback(self, callback: callable):
        """Register callback to be notified of state updates"""
        self.update_callbacks.append(callback)
    
    def _save_patient_state(self, patient_id: str):
        """Persist patient state to disk"""
        filepath = self.storage_path / f"{patient_id}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.patient_states[patient_id], f)
    
    def _load_patient_state(self, patient_id: str) -> bool:
        """Load patient state from disk"""
        filepath = self.storage_path / f"{patient_id}.pkl"
        
        if not filepath.exists():
            return False
        
        with open(filepath, 'rb') as f:
            self.patient_states[patient_id] = pickle.load(f)
        
        return True
    
    def export_patient_timeline(self, patient_id: str, 
                               format: str = 'csv') -> str:
        """Export patient timeline to file"""
        df = self.get_all_parameters_trajectory(patient_id)
        
        if format == 'csv':
            filepath = self.storage_path / f"{patient_id}_timeline.csv"
            df.to_csv(filepath)
        elif format == 'json':
            filepath = self.storage_path / f"{patient_id}_timeline.json"
            df.to_json(filepath, orient='records', date_format='iso')
        
        logger.info(f"✓ Exported timeline: {filepath}")
        return str(filepath)
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        total_states = sum(len(states) for states in self.patient_states.values())
        
        return {
            'total_patients': len(self.patient_states),
            'active_patients': len(self.active_patients),
            'total_state_snapshots': total_states,
            'avg_snapshots_per_patient': total_states / len(self.patient_states) if self.patient_states else 0
        }


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PATIENT STATE ENGINE DEMONSTRATION")
    print("="*80)
    
    # Initialize engine
    engine = PatientStateEngine()
    
    # Create initial patient state
    initial_state = PatientStateSnapshot(
        timestamp=datetime(2024, 1, 1),
        patient_id='P001',
        age=55,
        gender='male',
        ethnicity='caucasian',
        weight_kg=95,
        height_cm=175,
        bmi=31,
        waist_cm=105,
        systolic_bp=145,
        diastolic_bp=92,
        heart_rate=78,
        temperature_c=36.8,
        glucose_mmol_l=6.2,
        hba1c_percent=6.0,
        total_chol_mmol_l=5.8,
        ldl_mmol_l=4.2,
        hdl_mmol_l=1.0,
        triglycerides_mmol_l=2.5,
        creatinine_umol_l=90,
        egfr=85,
        alt=35,
        ast=30,
        insulin_resistance_index=1.5,
        beta_cell_function_percent=85,
        exercise_hours_week=1.0,
        smoking_status='current',
        stress_level=0.7
    )
    
    # Register patient
    engine.register_patient('P001', initial_state)
    
    # Simulate 5 years forward
    print("\nSimulating 5 years forward...")
    for year in range(1, 6):
        future_state = engine.simulate_forward('P001', days=365)
        engine.update_state('P001', future_state, source='simulation')
        print(f"Year {year}: BMI={future_state.bmi:.1f}, HbA1c={future_state.hba1c_percent:.1f}%, "
              f"BP={future_state.systolic_bp:.0f}/{future_state.diastolic_bp:.0f}")
    
    # Get trajectory
    print("\nHbA1c Trajectory:")
    hba1c_traj = engine.get_parameter_trajectory('P001', 'hba1c_percent')
    print(hba1c_traj)
    
    # Detect changes
    changes = engine.detect_state_changes('P001', lookback_days=1825)  # 5 years
    print(f"\nDetected {len(changes)} significant changes:")
    for change in changes:
        print(f"  • {change['parameter']}: {change['change']} ({change['significance']})")
    
    # Statistics
    stats = engine.get_statistics()
    print(f"\nEngine Statistics:")
    print(f"  Total patients: {stats['total_patients']}")
    print(f"  Total snapshots: {stats['total_state_snapshots']}")
    print(f"  Avg snapshots/patient: {stats['avg_snapshots_per_patient']:.1f}")
    
    # Export
    filepath = engine.export_patient_timeline('P001', format='csv')
    print(f"\n✓ Timeline exported to: {filepath}")
    
    print("\n" + "="*80)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*80)
