"""
Hybrid Digital Twin Model
Combines:
1. Literature-based physiological structure (organ agents)
2. Data-calibrated parameters (from MIMIC/real patients)
3. Deep learning predictions (LSTM for individual patterns)
"""

import numpy as np
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirofish_engine.parallel_digital_patient import ParallelDigitalPatient
from models.lstm_predictor import get_patient_predictor
from data_pipeline.mimic_data_loader import get_mimic_loader


class HybridDigitalTwin:
    """
    Evidence-based digital twin combining:
    - Physiological organ models (structure from literature)
    - Empirical parameters (calibrated from patient data)
    - ML predictions (LSTM for complex patterns)
    """
    
    def __init__(self, patient_id: str, seed_info: Dict[str, Any]):
        self.patient_id = patient_id
        self.seed_info = seed_info
        
        # Load empirical parameters from real patient data
        self.empirical_params = self._load_empirical_parameters()
        
        # Initialize parametric organ model
        self.organ_model = ParallelDigitalPatient(patient_id, seed_info)
        
        # Load LSTM predictor (if trained)
        self.lstm_predictor = None
        try:
            self.lstm_predictor = get_patient_predictor()
            self.lstm_predictor.load_model()
            print("✓ LSTM predictor loaded")
        except:
            print("⚠️  LSTM predictor not trained yet - using parametric model only")
    
    def _load_empirical_parameters(self) -> Dict[str, float]:
        """
        Load parameters calibrated from real patient data
        Replaces arbitrary values with evidence-based ones
        """
        # Try to load from MIMIC data
        try:
            loader = get_mimic_loader()
            loader.download_mimic()
            params = loader.calculate_empirical_decline_rates()
            print("✓ Loaded empirical parameters from patient data")
            return params
        except:
            print("⚠️  Using literature-based parameters (MIMIC data not available)")
            return self._get_literature_parameters()
    
    def _get_literature_parameters(self) -> Dict[str, float]:
        """
        Fallback: Use parameters from medical literature
        All values cited with sources
        """
        return {
            # eGFR decline rates
            # Source: KDIGO 2012, Coresh et al. JASN 2014
            'egfr_decline_per_day_mean': -1.0 / 365,  # Normal aging: -1 mL/min/year
            'egfr_decline_ckd_per_day': -4.0 / 365,   # CKD: -3 to -5 mL/min/year
            
            # HbA1c progression
            # Source: UKPDS 35, Diabetes Care 2001
            'hba1c_increase_per_day_mean': 0.2 / 365,  # Untreated diabetes: +0.2%/year
            'hba1c_increase_prediabetes': 0.05 / 365,  # Prediabetes: +0.05%/year
            
            # Beta cell function decline
            # Source: UKPDS 16, Diabetes 1995
            'beta_cell_decline_per_year': 0.04,  # 4% per year in type 2 diabetes
            
            # Insulin sensitivity decline
            # Source: DeFronzo RA, Diabetes 2004
            'insulin_sensitivity_decline_per_year': 0.02,  # 2% per year with obesity
            
            # Atherosclerosis progression
            # Source: Libby P, Nature 2002
            'atherosclerosis_increase_per_year': 0.01,  # 1% increase in plaque burden/year
            
            # Blood pressure increase with age
            # Source: Franklin SS, Circulation 1997
            'sbp_increase_per_year': 0.5,  # 0.5 mmHg/year after age 50
            
            # Exercise effects on insulin sensitivity
            # Source: Hawley JA, Diabetologia 2014
            'exercise_insulin_sensitivity_improvement': 0.15,  # 15% improvement with regular exercise
            
            # Diet effects on glucose
            # Source: Evert AB, Diabetes Care 2019
            'mediterranean_diet_hba1c_reduction': 0.3,  # 0.3% HbA1c reduction
        }
    
    def simulate_with_hybrid_model(
        self,
        lifestyle_inputs: Dict[str, Any],
        days: int = 1825
    ) -> Dict[str, Any]:
        """
        Run simulation using hybrid approach:
        1. Parametric organ model (physiological structure)
        2. Empirical parameters (from real data)
        3. LSTM correction (for individual patterns)
        """
        print(f"\n🔬 Running hybrid simulation for {days} days...")
        
        # Update organ model with empirical parameters
        self._update_organ_parameters()
        
        # Run parametric simulation
        timeline = []
        lstm_history = []
        
        for day in range(days):
            # Get daily lifestyle inputs
            daily_inputs = lifestyle_inputs.get(day, {})
            self.organ_model.environment.external_inputs.update(daily_inputs)
            
            # Parametric organ model step
            day_state = self.organ_model._simulate_one_day()
            timeline.append(day_state)
            
            # Collect history for LSTM
            if 'agents' in day_state:
                metabolic = day_state['agents'].get('metabolic', {}).get('state', {})
                cardiovascular = day_state['agents'].get('cardiovascular', {}).get('state', {})
                renal = day_state['agents'].get('renal', {}).get('state', {})
                
                lstm_history.append([
                    metabolic.get('glucose', 5.0),
                    metabolic.get('hba1c', 5.0),
                    renal.get('egfr', 90),
                    renal.get('creatinine', 80),
                    cardiovascular.get('systolic_bp', 120),
                    cardiovascular.get('diastolic_bp', 80)
                ])
            
            # LSTM correction every 30 days
            if self.lstm_predictor and day > 0 and day % 30 == 0 and len(lstm_history) >= 30:
                lstm_correction = self._apply_lstm_correction(lstm_history[-30:])
                if lstm_correction:
                    # Apply correction to organ states
                    self._apply_correction_to_organs(lstm_correction)
            
            # Check disease emergence
            emerged_diseases = self.organ_model._detect_disease_emergence()
            for disease in emerged_diseases:
                if disease.name not in [d.name for d in self.organ_model.diseases_emerged]:
                    self.organ_model.diseases_emerged.append(disease)
                    print(f"  ⚠️  Day {day}: {disease.name} emerged ({disease.probability:.0%})")
            
            if day % 365 == 0 and day > 0:
                print(f"  ✓ Year {day//365} complete")
        
        self.organ_model.timeline = timeline
        self.organ_model.current_day = days
        
        return {
            'timeline': timeline,
            'diseases_emerged': self.organ_model.diseases_emerged,
            'final_state': self.organ_model.agents
        }
    
    def _update_organ_parameters(self):
        """
        Update organ agents with empirical parameters
        Replaces arbitrary values with data-calibrated ones
        """
        params = self.empirical_params
        
        # Update metabolic agent
        metabolic = self.organ_model.agents.get('metabolic')
        if metabolic:
            # Store empirical decline rates as agent properties
            metabolic.hba1c_progression_rate = params.get('hba1c_increase_per_day_mean', 0.2/365)
            metabolic.beta_cell_decline_rate = params.get('beta_cell_decline_per_year', 0.04) / 365
        
        # Update renal agent
        renal = self.organ_model.agents.get('renal')
        if renal:
            renal.egfr_decline_rate = params.get('egfr_decline_per_day_mean', -1.0/365)
        
        print("✓ Organ parameters updated with empirical values")
    
    def _apply_lstm_correction(self, history: List[List[float]]) -> Optional[np.ndarray]:
        """
        Use LSTM to predict next values and calculate correction
        """
        try:
            history_array = np.array(history)
            predictions = self.lstm_predictor.predict(history_array)
            
            # Return predicted next value
            return predictions[0]  # First prediction
        except Exception as e:
            print(f"⚠️  LSTM correction failed: {e}")
            return None
    
    def _apply_correction_to_organs(self, correction: np.ndarray):
        """
        Apply LSTM correction to organ states
        Blend parametric model with ML prediction
        """
        # Correction is: [glucose, hba1c, egfr, creatinine, sbp, dbp]
        blend_factor = 0.3  # 30% ML, 70% parametric
        
        metabolic = self.organ_model.agents.get('metabolic')
        if metabolic and len(correction) >= 2:
            current_glucose = metabolic.state.get('glucose', 5.0)
            current_hba1c = metabolic.state.get('hba1c', 5.0)
            
            # Blend predictions
            metabolic.state['glucose'] = (1 - blend_factor) * current_glucose + blend_factor * correction[0]
            metabolic.state['hba1c'] = (1 - blend_factor) * current_hba1c + blend_factor * correction[1]
        
        renal = self.organ_model.agents.get('renal')
        if renal and len(correction) >= 4:
            current_egfr = renal.state.get('egfr', 90)
            renal.state['egfr'] = (1 - blend_factor) * current_egfr + blend_factor * correction[2]
        
        cardiovascular = self.organ_model.agents.get('cardiovascular')
        if cardiovascular and len(correction) >= 6:
            current_sbp = cardiovascular.state.get('systolic_bp', 120)
            cardiovascular.state['systolic_bp'] = (1 - blend_factor) * current_sbp + blend_factor * correction[4]


def create_hybrid_twin(patient_id: str, seed_info: Dict[str, Any]) -> HybridDigitalTwin:
    """Factory function to create hybrid digital twin"""
    return HybridDigitalTwin(patient_id, seed_info)
