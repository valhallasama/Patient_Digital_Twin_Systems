"""
Markov Disease Progression Models
State-based disease progression using transition probabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiabetesState(Enum):
    """Diabetes disease states"""
    HEALTHY = 0
    PREDIABETES = 1
    DIABETES_CONTROLLED = 2
    DIABETES_UNCONTROLLED = 3
    DIABETES_WITH_MICROVASCULAR = 4
    DIABETES_WITH_MACROVASCULAR = 5
    DIABETES_WITH_ESRD = 6
    DEATH = 7


class CVDState(Enum):
    """Cardiovascular disease states"""
    HEALTHY = 0
    SUBCLINICAL_ATHEROSCLEROSIS = 1
    STABLE_ANGINA = 2
    UNSTABLE_ANGINA = 3
    MYOCARDIAL_INFARCTION = 4
    HEART_FAILURE = 5
    DEATH = 6


class CKDState(Enum):
    """Chronic kidney disease states"""
    NORMAL = 0  # eGFR >= 90
    MILD_DECREASE = 1  # eGFR 60-89
    MODERATE_DECREASE = 2  # eGFR 30-59
    SEVERE_DECREASE = 3  # eGFR 15-29
    KIDNEY_FAILURE = 4  # eGFR < 15
    DIALYSIS = 5
    DEATH = 6


@dataclass
class TransitionProbabilities:
    """Transition probability matrix for Markov model"""
    states: List[str]
    matrix: np.ndarray
    time_unit: str = "year"  # per year, per month, etc.
    
    def get_probability(self, from_state: int, to_state: int) -> float:
        """Get transition probability from one state to another"""
        return self.matrix[from_state, to_state]
    
    def validate(self) -> bool:
        """Validate that rows sum to 1"""
        row_sums = self.matrix.sum(axis=1)
        return np.allclose(row_sums, 1.0)


class MarkovDiseaseModel:
    """
    Base class for Markov disease progression models
    """
    
    def __init__(self, states: Enum, transition_matrix: np.ndarray):
        self.states = states
        self.num_states = len(states)
        self.transition_matrix = transition_matrix
        
        # Validate transition matrix
        if not self._validate_transition_matrix():
            raise ValueError("Invalid transition matrix: rows must sum to 1")
    
    def _validate_transition_matrix(self) -> bool:
        """Ensure transition matrix is valid (rows sum to 1)"""
        row_sums = self.transition_matrix.sum(axis=1)
        return np.allclose(row_sums, 1.0, atol=1e-6)
    
    def simulate_trajectory(self, initial_state: int, time_steps: int, 
                          patient_factors: Optional[Dict] = None) -> List[int]:
        """
        Simulate disease trajectory using Markov chain
        
        Args:
            initial_state: Starting disease state
            time_steps: Number of time steps to simulate
            patient_factors: Patient-specific factors that modify transition probabilities
        
        Returns:
            List of states over time
        """
        trajectory = [initial_state]
        current_state = initial_state
        
        for step in range(time_steps):
            # Get transition probabilities for current state
            trans_probs = self.transition_matrix[current_state, :].copy()
            
            # Modify probabilities based on patient factors
            if patient_factors:
                trans_probs = self._modify_probabilities(current_state, trans_probs, patient_factors)
                trans_probs = trans_probs / trans_probs.sum()  # Renormalize
            
            # Sample next state
            next_state = np.random.choice(self.num_states, p=trans_probs)
            trajectory.append(next_state)
            current_state = next_state
            
            # Stop if reached absorbing state (death)
            if self._is_absorbing_state(current_state):
                break
        
        return trajectory
    
    def _modify_probabilities(self, current_state: int, probs: np.ndarray, 
                             factors: Dict) -> np.ndarray:
        """Modify transition probabilities based on patient factors"""
        # Override in subclasses
        return probs
    
    def _is_absorbing_state(self, state: int) -> bool:
        """Check if state is absorbing (e.g., death)"""
        # State is absorbing if P(state -> state) = 1
        return self.transition_matrix[state, state] == 1.0
    
    def calculate_state_probabilities(self, initial_state: int, 
                                     time_steps: int) -> np.ndarray:
        """
        Calculate probability distribution over states at each time step
        Using matrix multiplication: P(t) = P(0) * T^t
        """
        # Initial distribution
        p0 = np.zeros(self.num_states)
        p0[initial_state] = 1.0
        
        # Store probabilities over time
        probs_over_time = np.zeros((time_steps + 1, self.num_states))
        probs_over_time[0] = p0
        
        # Compute P(t) = P(0) * T^t
        current_probs = p0
        for t in range(1, time_steps + 1):
            current_probs = current_probs @ self.transition_matrix
            probs_over_time[t] = current_probs
        
        return probs_over_time


class DiabetesMarkovModel(MarkovDiseaseModel):
    """
    Markov model for diabetes progression
    
    States:
    0: Healthy
    1: Prediabetes
    2: Diabetes (controlled)
    3: Diabetes (uncontrolled)
    4: Diabetes with microvascular complications
    5: Diabetes with macrovascular complications
    6: ESRD (End-stage renal disease)
    7: Death
    """
    
    def __init__(self):
        # Transition matrix (annual probabilities)
        # Rows: current state, Columns: next state
        transition_matrix = np.array([
            # H    Pre   DC   DU   Micro Macro ESRD Death
            [0.90, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01],  # Healthy
            [0.05, 0.80, 0.10, 0.03, 0.00, 0.00, 0.00, 0.02],  # Prediabetes
            [0.00, 0.00, 0.85, 0.10, 0.03, 0.01, 0.00, 0.01],  # Diabetes controlled
            [0.00, 0.00, 0.05, 0.75, 0.10, 0.05, 0.02, 0.03],  # Diabetes uncontrolled
            [0.00, 0.00, 0.00, 0.00, 0.85, 0.05, 0.05, 0.05],  # Microvascular
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.05, 0.10],  # Macrovascular
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.20],  # ESRD
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # Death (absorbing)
        ])
        
        super().__init__(DiabetesState, transition_matrix)
    
    def _modify_probabilities(self, current_state: int, probs: np.ndarray, 
                             factors: Dict) -> np.ndarray:
        """Modify transition probabilities based on patient risk factors"""
        
        # Risk factors
        age = factors.get('age', 50)
        bmi = factors.get('bmi', 25)
        hba1c = factors.get('hba1c', 5.5)
        exercise = factors.get('exercise_hours', 0)
        on_medication = factors.get('on_metformin', False)
        
        # Age effect (older = faster progression)
        age_multiplier = 1.0 + (age - 50) / 100
        
        # BMI effect
        bmi_multiplier = 1.0 + max(0, (bmi - 25) / 20)
        
        # HbA1c effect
        hba1c_multiplier = 1.0 + max(0, (hba1c - 5.5) / 5)
        
        # Exercise protective effect
        exercise_multiplier = 1.0 - min(0.3, exercise / 20)
        
        # Medication effect
        medication_multiplier = 0.7 if on_medication else 1.0
        
        # Apply multipliers to progression probabilities
        if current_state == DiabetesState.HEALTHY.value:
            # Increase prediabetes risk
            probs[DiabetesState.PREDIABETES.value] *= age_multiplier * bmi_multiplier * exercise_multiplier
        
        elif current_state == DiabetesState.PREDIABETES.value:
            # Increase diabetes risk
            probs[DiabetesState.DIABETES_CONTROLLED.value] *= bmi_multiplier * hba1c_multiplier * exercise_multiplier
            probs[DiabetesState.DIABETES_UNCONTROLLED.value] *= bmi_multiplier * hba1c_multiplier
        
        elif current_state == DiabetesState.DIABETES_CONTROLLED.value:
            # Medication reduces progression
            probs[DiabetesState.DIABETES_UNCONTROLLED.value] *= medication_multiplier
            probs[DiabetesState.DIABETES_WITH_MICROVASCULAR.value] *= medication_multiplier
        
        return probs


class CVDMarkovModel(MarkovDiseaseModel):
    """
    Markov model for cardiovascular disease progression
    
    States:
    0: Healthy
    1: Subclinical atherosclerosis
    2: Stable angina
    3: Unstable angina
    4: Myocardial infarction
    5: Heart failure
    6: Death
    """
    
    def __init__(self):
        transition_matrix = np.array([
            # H    Sub   SA   UA   MI   HF   Death
            [0.92, 0.06, 0.01, 0.00, 0.00, 0.00, 0.01],  # Healthy
            [0.00, 0.85, 0.10, 0.03, 0.01, 0.00, 0.01],  # Subclinical
            [0.00, 0.00, 0.80, 0.10, 0.05, 0.03, 0.02],  # Stable angina
            [0.00, 0.00, 0.05, 0.60, 0.25, 0.05, 0.05],  # Unstable angina
            [0.00, 0.00, 0.00, 0.00, 0.70, 0.20, 0.10],  # MI
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.15],  # Heart failure
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # Death
        ])
        
        super().__init__(CVDState, transition_matrix)
    
    def _modify_probabilities(self, current_state: int, probs: np.ndarray, 
                             factors: Dict) -> np.ndarray:
        """Modify based on CVD risk factors"""
        
        age = factors.get('age', 50)
        systolic_bp = factors.get('systolic_bp', 120)
        ldl = factors.get('ldl', 3.0)
        smoking = factors.get('smoking', False)
        diabetes = factors.get('has_diabetes', False)
        on_statin = factors.get('on_statin', False)
        
        # Risk multipliers
        age_mult = 1.0 + (age - 50) / 50
        bp_mult = 1.0 + max(0, (systolic_bp - 120) / 40)
        ldl_mult = 1.0 + max(0, (ldl - 3.0) / 2.0)
        smoking_mult = 2.0 if smoking else 1.0
        diabetes_mult = 1.5 if diabetes else 1.0
        statin_mult = 0.7 if on_statin else 1.0
        
        # Apply to progression
        if current_state == CVDState.HEALTHY.value:
            probs[CVDState.SUBCLINICAL_ATHEROSCLEROSIS.value] *= (
                age_mult * bp_mult * ldl_mult * smoking_mult * diabetes_mult * statin_mult
            )
        
        elif current_state == CVDState.SUBCLINICAL_ATHEROSCLEROSIS.value:
            probs[CVDState.STABLE_ANGINA.value] *= bp_mult * ldl_mult * statin_mult
            probs[CVDState.UNSTABLE_ANGINA.value] *= smoking_mult * diabetes_mult
        
        return probs


class CKDMarkovModel(MarkovDiseaseModel):
    """
    Markov model for chronic kidney disease progression
    Based on eGFR stages
    """
    
    def __init__(self):
        transition_matrix = np.array([
            # Normal Mild  Mod  Sev  Fail Dial Death
            [0.95,  0.04, 0.00, 0.00, 0.00, 0.00, 0.01],  # Normal
            [0.00,  0.90, 0.08, 0.01, 0.00, 0.00, 0.01],  # Mild
            [0.00,  0.00, 0.85, 0.10, 0.03, 0.00, 0.02],  # Moderate
            [0.00,  0.00, 0.00, 0.80, 0.10, 0.05, 0.05],  # Severe
            [0.00,  0.00, 0.00, 0.00, 0.70, 0.20, 0.10],  # Failure
            [0.00,  0.00, 0.00, 0.00, 0.00, 0.85, 0.15],  # Dialysis
            [0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # Death
        ])
        
        super().__init__(CKDState, transition_matrix)
    
    def _modify_probabilities(self, current_state: int, probs: np.ndarray, 
                             factors: Dict) -> np.ndarray:
        """Modify based on CKD risk factors"""
        
        diabetes = factors.get('has_diabetes', False)
        hypertension = factors.get('has_hypertension', False)
        age = factors.get('age', 50)
        
        diabetes_mult = 2.0 if diabetes else 1.0
        htn_mult = 1.5 if hypertension else 1.0
        age_mult = 1.0 + (age - 50) / 100
        
        # Apply to all progression transitions
        for i in range(len(probs)):
            if i > current_state and i < len(probs) - 1:  # Progression states
                probs[i] *= diabetes_mult * htn_mult * age_mult
        
        return probs


class MultiDiseaseMarkovSimulator:
    """
    Simulate multiple diseases simultaneously with interactions
    """
    
    def __init__(self):
        self.diabetes_model = DiabetesMarkovModel()
        self.cvd_model = CVDMarkovModel()
        self.ckd_model = CKDMarkovModel()
    
    def simulate_patient(self, initial_states: Dict[str, int], 
                        patient_factors: Dict, years: int = 10) -> pd.DataFrame:
        """
        Simulate multiple disease trajectories with interactions
        
        Args:
            initial_states: {'diabetes': 0, 'cvd': 0, 'ckd': 0}
            patient_factors: Patient characteristics
            years: Simulation duration
        
        Returns:
            DataFrame with disease states over time
        """
        
        results = {
            'year': [],
            'diabetes_state': [],
            'cvd_state': [],
            'ckd_state': []
        }
        
        # Current states
        diabetes_state = initial_states.get('diabetes', 0)
        cvd_state = initial_states.get('cvd', 0)
        ckd_state = initial_states.get('ckd', 0)
        
        for year in range(years + 1):
            # Record current states
            results['year'].append(year)
            results['diabetes_state'].append(diabetes_state)
            results['cvd_state'].append(cvd_state)
            results['ckd_state'].append(ckd_state)
            
            if year < years:
                # Update patient factors based on current disease states
                factors = patient_factors.copy()
                factors['has_diabetes'] = diabetes_state >= DiabetesState.DIABETES_CONTROLLED.value
                factors['has_hypertension'] = cvd_state >= CVDState.SUBCLINICAL_ATHEROSCLEROSIS.value
                
                # Simulate next year
                diabetes_traj = self.diabetes_model.simulate_trajectory(
                    diabetes_state, 1, factors
                )
                cvd_traj = self.cvd_model.simulate_trajectory(
                    cvd_state, 1, factors
                )
                ckd_traj = self.ckd_model.simulate_trajectory(
                    ckd_state, 1, factors
                )
                
                diabetes_state = diabetes_traj[-1]
                cvd_state = cvd_traj[-1]
                ckd_state = ckd_traj[-1]
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("MARKOV DISEASE PROGRESSION MODELS")
    print("="*80)
    
    # Test diabetes model
    print("\n1. DIABETES PROGRESSION")
    print("-" * 80)
    
    diabetes_model = DiabetesMarkovModel()
    
    patient_factors = {
        'age': 55,
        'bmi': 32,
        'hba1c': 6.0,
        'exercise_hours': 2,
        'on_metformin': False
    }
    
    trajectory = diabetes_model.simulate_trajectory(
        initial_state=DiabetesState.PREDIABETES.value,
        time_steps=10,
        patient_factors=patient_factors
    )
    
    print(f"Initial state: {DiabetesState(trajectory[0]).name}")
    print(f"10-year trajectory:")
    for year, state in enumerate(trajectory):
        print(f"  Year {year}: {DiabetesState(state).name}")
    
    # Test multi-disease simulation
    print("\n2. MULTI-DISEASE SIMULATION")
    print("-" * 80)
    
    simulator = MultiDiseaseMarkovSimulator()
    
    initial_states = {
        'diabetes': DiabetesState.PREDIABETES.value,
        'cvd': CVDState.HEALTHY.value,
        'ckd': CKDState.NORMAL.value
    }
    
    patient_factors = {
        'age': 55,
        'bmi': 32,
        'hba1c': 6.0,
        'systolic_bp': 145,
        'ldl': 4.2,
        'smoking': True,
        'exercise_hours': 1,
        'on_metformin': False,
        'on_statin': False
    }
    
    results = simulator.simulate_patient(initial_states, patient_factors, years=10)
    
    print("\nPatient trajectory (10 years):")
    print(results.to_string(index=False))
    
    # Calculate state probabilities
    print("\n3. PROBABILITY DISTRIBUTION OVER TIME")
    print("-" * 80)
    
    probs = diabetes_model.calculate_state_probabilities(
        initial_state=DiabetesState.PREDIABETES.value,
        time_steps=10
    )
    
    print("\nProbability of each diabetes state over 10 years:")
    print("Year | Healthy | Prediab | Control | Uncontr | Micro | Macro | ESRD | Death")
    print("-" * 80)
    for year in range(0, 11, 2):
        probs_str = " | ".join([f"{p:6.1%}" for p in probs[year]])
        print(f"{year:4d} | {probs_str}")
