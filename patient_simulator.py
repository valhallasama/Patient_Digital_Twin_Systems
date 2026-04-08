#!/usr/bin/env python3
"""
Single Patient Digital Twin Simulator

YOUR VISION IMPLEMENTED:
1. Upload patient health data (text + test results)
2. Create digital twin from this information
3. Simulate 5-10 years of organ changes
4. Predict disease risks with probabilities
5. Recommend interventions to reduce risks
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from graph_learning.stateful_organ_agents import MultiOrganSimulator, OrganState
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid
from graph_learning.organ_gnn import OrganGraphNetwork
from graph_learning.temporal_transformer import TemporalTransformerEncoder


@dataclass
class PatientData:
    """Structured patient data"""
    # Demographics
    age: int
    sex: str  # 'male' or 'female'
    
    # Lifestyle (parsed from text or structured)
    occupation: str = "unknown"
    exercise_hours_per_week: float = 0.0
    sleep_hours_per_night: float = 7.0
    smoking: bool = False
    alcohol_drinks_per_week: float = 0.0
    stress_level: float = 0.5  # 0-1 scale
    diet_quality: str = "moderate"  # poor/moderate/good
    
    # Test results (lab values)
    bmi: float = 25.0
    systolic_bp: float = 120.0
    diastolic_bp: float = 80.0
    fasting_glucose: float = 100.0
    hba1c: float = 5.5
    total_cholesterol: float = 200.0
    ldl: float = 100.0
    hdl: float = 50.0
    triglycerides: float = 150.0
    creatinine: float = 1.0
    egfr: float = 90.0
    alt: float = 25.0
    ast: float = 25.0
    crp: float = 1.0
    hemoglobin: float = 14.0
    
    # Medical history
    existing_conditions: List[str] = None
    medications: List[str] = None
    family_history: List[str] = None
    
    def __post_init__(self):
        if self.existing_conditions is None:
            self.existing_conditions = []
        if self.medications is None:
            self.medications = []
        if self.family_history is None:
            self.family_history = []


class NaturalLanguageParser:
    """Parse natural language patient descriptions"""
    
    @staticmethod
    def parse_lifestyle_description(text: str) -> Dict:
        """
        Parse lifestyle from natural language
        
        Example: "sedentary office worker, smoker, drinks 3 beers daily, 
                  sleeps 5 hours, high stress, eats fast food"
        """
        text = text.lower()
        
        lifestyle = {
            'occupation': 'unknown',
            'exercise_hours_per_week': 0.0,
            'sleep_hours_per_night': 7.0,
            'smoking': False,
            'alcohol_drinks_per_week': 0.0,
            'stress_level': 0.5,
            'diet_quality': 'moderate'
        }
        
        # Occupation
        if 'office' in text or 'desk' in text or 'sedentary' in text:
            lifestyle['occupation'] = 'office_worker'
        elif 'manual' in text or 'labor' in text or 'construction' in text:
            lifestyle['occupation'] = 'manual_labor'
        elif 'healthcare' in text or 'nurse' in text or 'doctor' in text:
            lifestyle['occupation'] = 'healthcare'
        
        # Exercise
        if 'sedentary' in text or 'no exercise' in text or 'inactive' in text:
            lifestyle['exercise_hours_per_week'] = 0.0
        elif 'active' in text or 'exercise' in text:
            # Try to extract hours
            exercise_match = re.search(r'(\d+)\s*(hour|hr|h)', text)
            if exercise_match:
                lifestyle['exercise_hours_per_week'] = float(exercise_match.group(1))
            else:
                lifestyle['exercise_hours_per_week'] = 3.0  # Default active
        
        # Sleep
        sleep_match = re.search(r'sleep[s]?\s*(\d+)\s*(hour|hr|h)', text)
        if sleep_match:
            lifestyle['sleep_hours_per_night'] = float(sleep_match.group(1))
        elif 'poor sleep' in text or 'insomnia' in text:
            lifestyle['sleep_hours_per_night'] = 5.0
        
        # Smoking
        if 'smok' in text or 'cigarette' in text:
            lifestyle['smoking'] = True
        
        # Alcohol
        if 'drink' in text or 'alcohol' in text or 'beer' in text or 'wine' in text:
            # Try to extract amount
            alcohol_match = re.search(r'(\d+)\s*(drink|beer|glass|bottle)', text)
            if alcohol_match:
                drinks_per_day = float(alcohol_match.group(1))
                lifestyle['alcohol_drinks_per_week'] = drinks_per_day * 7
            else:
                lifestyle['alcohol_drinks_per_week'] = 7.0  # Default moderate
        
        # Stress
        if 'high stress' in text or 'stressed' in text or 'anxiety' in text:
            lifestyle['stress_level'] = 0.8
        elif 'low stress' in text or 'relaxed' in text:
            lifestyle['stress_level'] = 0.2
        
        # Diet
        if 'fast food' in text or 'poor diet' in text or 'junk food' in text:
            lifestyle['diet_quality'] = 'poor'
        elif 'healthy' in text or 'vegetables' in text or 'balanced' in text:
            lifestyle['diet_quality'] = 'good'
        
        return lifestyle
    
    @staticmethod
    def parse_patient_description(
        description: str,
        age: int,
        sex: str,
        test_results: Dict
    ) -> PatientData:
        """
        Parse full patient description
        
        Args:
            description: Natural language description of lifestyle
            age: Patient age
            sex: 'male' or 'female'
            test_results: Dictionary of lab values
        """
        lifestyle = NaturalLanguageParser.parse_lifestyle_description(description)
        
        return PatientData(
            age=age,
            sex=sex,
            **lifestyle,
            **test_results
        )


@dataclass
class SimulationResult:
    """Results from patient simulation"""
    trajectory: List[Dict[str, OrganState]]
    disease_risks: Dict[str, float]
    disease_onset_times: Dict[str, float]
    interventions: List[Dict]
    attention_weights: torch.Tensor
    
    def get_summary(self) -> str:
        """Generate human-readable summary"""
        summary = []
        summary.append("=" * 80)
        summary.append("DIGITAL TWIN SIMULATION RESULTS")
        summary.append("=" * 80)
        
        summary.append("\n📊 DISEASE RISK PREDICTIONS (10-year outlook):\n")
        
        # Sort by risk
        sorted_risks = sorted(
            self.disease_risks.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for disease, risk in sorted_risks[:10]:  # Top 10
            if risk > 0.1:  # Only show >10% risk
                onset = self.disease_onset_times.get(disease, 120)
                risk_level = "🔴 HIGH" if risk > 0.5 else "🟡 MODERATE" if risk > 0.3 else "🟢 LOW"
                summary.append(
                    f"  {risk_level} {disease.replace('_', ' ').title()}: "
                    f"{risk*100:.1f}% risk (onset ~{onset:.0f} months)"
                )
        
        summary.append("\n💡 RECOMMENDED INTERVENTIONS:\n")
        for i, intervention in enumerate(self.interventions[:5], 1):
            summary.append(
                f"  {i}. {intervention['action']}: "
                f"Reduces {intervention['disease']} risk by {intervention['reduction']*100:.1f}%"
            )
        
        return "\n".join(summary)


class PatientDigitalTwin:
    """
    Digital Twin for a single patient
    
    YOUR VISION IMPLEMENTED HERE!
    """
    
    def __init__(
        self,
        patient_data: PatientData,
        model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.patient = patient_data
        self.device = device
        
        # Load trained model
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            # Create untrained model (for demo)
            self._create_demo_model()
        
        # Initialize simulator
        self.simulator = MultiOrganSimulator(
            organ_configs=self.organ_dims,
            gnn_model=self.gnn,
            transformer_model=self.transformer,
            hidden_dim=64,
            use_stochastic=True
        )
    
    def _create_demo_model(self):
        """Create model for demo (before training completes)"""
        self.organ_dims = {
            'metabolic': 4,
            'cardiovascular': 5,
            'liver': 2,
            'kidney': 2,
            'immune': 1,
            'neural': 1,
            'lifestyle': 4
        }
        
        self.gnn = OrganGraphNetwork(
            node_feature_dims=self.organ_dims,
            hidden_dim=64,
            num_gat_layers=2
        )
        
        self.transformer = TemporalTransformerEncoder(
            d_model=128,
            num_heads=8,
            num_layers=4,
            num_organs=7
        )
        
        self.disease_predictor = GNNTransformerHybrid(
            node_feature_dims=self.organ_dims,
            gnn_hidden_dim=64,
            transformer_d_model=512,
            transformer_num_heads=8,
            transformer_num_layers=4,
            num_diseases=24
        )
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        # Load model weights
        # TODO: Implement after training completes
        pass
    
    def _extract_organ_features(self) -> Dict[str, torch.Tensor]:
        """Extract organ features from patient data"""
        p = self.patient
        
        features = {
            'metabolic': torch.tensor([
                p.fasting_glucose,
                p.hba1c,
                p.bmi,
                p.bmi * 2.5  # Approximate waist circumference
            ], dtype=torch.float32).unsqueeze(0),
            
            'cardiovascular': torch.tensor([
                p.systolic_bp,
                p.diastolic_bp,
                p.ldl,
                p.hdl,
                p.triglycerides
            ], dtype=torch.float32).unsqueeze(0),
            
            'liver': torch.tensor([
                p.alt,
                p.ast
            ], dtype=torch.float32).unsqueeze(0),
            
            'kidney': torch.tensor([
                p.creatinine,
                p.egfr
            ], dtype=torch.float32).unsqueeze(0),
            
            'immune': torch.tensor([
                p.crp
            ], dtype=torch.float32).unsqueeze(0),
            
            'neural': torch.tensor([
                p.stress_level
            ], dtype=torch.float32).unsqueeze(0),
            
            'lifestyle': torch.tensor([
                p.exercise_hours_per_week / 20.0,  # Normalize
                1.0 if p.smoking else 0.0,
                p.alcohol_drinks_per_week / 20.0,
                p.sleep_hours_per_night / 10.0
            ], dtype=torch.float32).unsqueeze(0)
        }
        
        return features
    
    def simulate(
        self,
        years: int = 10,
        stochastic: bool = True,
        num_simulations: int = 1
    ) -> SimulationResult:
        """
        Simulate patient's future health trajectory
        
        Args:
            years: Number of years to simulate
            stochastic: Include biological variability
            num_simulations: Number of stochastic runs (for uncertainty)
        
        Returns:
            SimulationResult with predictions and recommendations
        """
        num_months = years * 12
        
        # Extract initial features
        initial_features = self._extract_organ_features()
        
        # Create edge index (organ graph)
        num_organs = len(self.organ_dims)
        edge_index = torch.tensor([
            [i, j] for i in range(num_organs) 
            for j in range(num_organs) if i != j
        ]).t()
        
        # Run simulation
        trajectory, attention_info = self.simulator.simulate_trajectory(
            initial_features=initial_features,
            edge_index=edge_index,
            num_steps=num_months,
            stochastic=stochastic
        )
        
        # Predict diseases from trajectory
        disease_risks, onset_times = self._predict_diseases(trajectory)
        
        # Generate intervention recommendations
        interventions = self._recommend_interventions(disease_risks)
        
        return SimulationResult(
            trajectory=trajectory,
            disease_risks=disease_risks,
            disease_onset_times=onset_times,
            interventions=interventions,
            attention_weights=attention_info.get('pooling_weights')
        )
    
    def _predict_diseases(
        self,
        trajectory: List[Dict[str, OrganState]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Predict disease risks from trajectory"""
        # Extract features at each timestep
        risks_over_time = []
        
        for states in trajectory:
            features = torch.cat([
                states[organ].features for organ in sorted(self.organ_dims.keys())
            ], dim=-1)
            
            # Predict (using demo logic for now)
            # TODO: Use trained model after training completes
            risks = self._demo_disease_prediction(states)
            risks_over_time.append(risks)
        
        # Aggregate risks
        final_risks = {}
        onset_times = {}
        
        for disease in GNNTransformerHybrid.DISEASE_NAMES:
            disease_trajectory = [r.get(disease, 0.0) for r in risks_over_time]
            final_risks[disease] = max(disease_trajectory)
            
            # Find onset time (when risk exceeds 50%)
            for t, risk in enumerate(disease_trajectory):
                if risk > 0.5:
                    onset_times[disease] = t
                    break
            else:
                onset_times[disease] = len(trajectory)
        
        return final_risks, onset_times
    
    def _demo_disease_prediction(self, states: Dict[str, OrganState]) -> Dict[str, float]:
        """Demo disease prediction (before training)"""
        # Simple rule-based predictions for demo
        metabolic = states['metabolic'].features[0]
        cardio = states['cardiovascular'].features[0]
        
        glucose = metabolic[0].item()
        hba1c = metabolic[1].item()
        bmi = metabolic[2].item()
        
        bp_sys = cardio[0].item()
        ldl = cardio[2].item()
        
        risks = {}
        
        # Diabetes risk
        diabetes_risk = 0.0
        if glucose > 100: diabetes_risk += 0.2
        if glucose > 126: diabetes_risk += 0.3
        if hba1c > 5.7: diabetes_risk += 0.2
        if hba1c > 6.5: diabetes_risk += 0.3
        if bmi > 30: diabetes_risk += 0.2
        risks['diabetes'] = min(diabetes_risk, 1.0)
        
        # Hypertension risk
        hypertension_risk = 0.0
        if bp_sys > 130: hypertension_risk += 0.3
        if bp_sys > 140: hypertension_risk += 0.4
        if bmi > 30: hypertension_risk += 0.2
        risks['hypertension'] = min(hypertension_risk, 1.0)
        
        # CVD risk
        cvd_risk = 0.0
        if bp_sys > 140: cvd_risk += 0.2
        if ldl > 130: cvd_risk += 0.2
        if self.patient.smoking: cvd_risk += 0.3
        if diabetes_risk > 0.5: cvd_risk += 0.2
        risks['cvd'] = min(cvd_risk, 1.0)
        
        # Fill in other diseases with low baseline risk
        for disease in GNNTransformerHybrid.DISEASE_NAMES:
            if disease not in risks:
                risks[disease] = 0.05
        
        return risks
    
    def _recommend_interventions(
        self,
        disease_risks: Dict[str, float]
    ) -> List[Dict]:
        """Recommend interventions to reduce risks"""
        interventions = []
        
        # Identify high-risk diseases
        high_risk_diseases = [
            disease for disease, risk in disease_risks.items()
            if risk > 0.3
        ]
        
        # Lifestyle interventions
        if self.patient.bmi > 30:
            for disease in ['diabetes', 'hypertension', 'cvd']:
                if disease in high_risk_diseases:
                    interventions.append({
                        'action': 'Lose 10% body weight (diet + exercise)',
                        'disease': disease,
                        'reduction': 0.25,
                        'timeframe': '6-12 months'
                    })
        
        if self.patient.smoking:
            for disease in ['cvd', 'copd', 'lung_cancer']:
                if disease in high_risk_diseases:
                    interventions.append({
                        'action': 'Quit smoking',
                        'disease': disease,
                        'reduction': 0.35,
                        'timeframe': 'Immediate'
                    })
        
        if self.patient.exercise_hours_per_week < 2:
            for disease in ['diabetes', 'cvd', 'hypertension']:
                if disease in high_risk_diseases:
                    interventions.append({
                        'action': 'Exercise 150 min/week (moderate intensity)',
                        'disease': disease,
                        'reduction': 0.20,
                        'timeframe': '3-6 months'
                    })
        
        if self.patient.alcohol_drinks_per_week > 14:
            interventions.append({
                'action': 'Reduce alcohol to <7 drinks/week',
                'disease': 'liver_disease',
                'reduction': 0.30,
                'timeframe': '3 months'
            })
        
        # Medical interventions
        if disease_risks.get('hypertension', 0) > 0.5:
            interventions.append({
                'action': 'Start antihypertensive medication',
                'disease': 'hypertension',
                'reduction': 0.40,
                'timeframe': 'Consult physician'
            })
        
        if disease_risks.get('diabetes', 0) > 0.5:
            interventions.append({
                'action': 'Start metformin (if prediabetic)',
                'disease': 'diabetes',
                'reduction': 0.30,
                'timeframe': 'Consult physician'
            })
        
        return interventions
    
    def test_intervention(
        self,
        intervention: str,
        years: int = 10
    ) -> Tuple[SimulationResult, SimulationResult]:
        """
        Test impact of an intervention
        
        Args:
            intervention: What to change (e.g., 'quit_smoking', 'lose_weight')
            years: Simulation duration
        
        Returns:
            (baseline_result, intervention_result)
        """
        # Baseline simulation
        baseline = self.simulate(years=years)
        
        # Modify patient data based on intervention
        modified_patient = PatientData(**self.patient.__dict__)
        
        if intervention == 'quit_smoking':
            modified_patient.smoking = False
        elif intervention == 'lose_weight':
            modified_patient.bmi *= 0.9  # 10% weight loss
        elif intervention == 'exercise':
            modified_patient.exercise_hours_per_week = 5.0
        elif intervention == 'reduce_alcohol':
            modified_patient.alcohol_drinks_per_week *= 0.5
        
        # Create new twin with modified data
        modified_twin = PatientDigitalTwin(modified_patient, device=self.device)
        intervention_result = modified_twin.simulate(years=years)
        
        return baseline, intervention_result


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("PATIENT DIGITAL TWIN SIMULATOR - YOUR VISION IMPLEMENTED!")
    print("=" * 80)
    
    # Example patient from natural language description
    patient_description = """
    Sedentary office worker, smoker (1 pack/day), drinks 3 beers daily,
    sleeps only 5 hours per night, high stress job, eats fast food regularly,
    no exercise
    """
    
    test_results = {
        'bmi': 32.0,
        'systolic_bp': 145.0,
        'diastolic_bp': 92.0,
        'fasting_glucose': 115.0,
        'hba1c': 5.9,
        'total_cholesterol': 245.0,
        'ldl': 165.0,
        'hdl': 38.0,
        'triglycerides': 210.0,
        'creatinine': 1.1,
        'alt': 42.0,
        'ast': 38.0,
        'crp': 3.2
    }
    
    # Parse patient data
    patient_data = NaturalLanguageParser.parse_patient_description(
        description=patient_description,
        age=45,
        sex='male',
        test_results=test_results
    )
    
    print("\n📋 PATIENT PROFILE:")
    print(f"  Age: {patient_data.age}, Sex: {patient_data.sex}")
    print(f"  Occupation: {patient_data.occupation}")
    print(f"  Smoking: {patient_data.smoking}")
    print(f"  Exercise: {patient_data.exercise_hours_per_week} hrs/week")
    print(f"  Sleep: {patient_data.sleep_hours_per_night} hrs/night")
    print(f"  BMI: {patient_data.bmi}")
    print(f"  Blood Pressure: {patient_data.systolic_bp}/{patient_data.diastolic_bp}")
    print(f"  Glucose: {patient_data.fasting_glucose}, HbA1c: {patient_data.hba1c}")
    
    # Create digital twin
    print("\n🔬 Creating digital twin...")
    twin = PatientDigitalTwin(patient_data)
    
    # Simulate 10 years
    print("\n⏳ Simulating 10-year health trajectory...")
    result = twin.simulate(years=10, stochastic=True)
    
    # Print results
    print("\n" + result.get_summary())
    
    # Test intervention
    print("\n\n🧪 TESTING INTERVENTION: Quit Smoking")
    print("=" * 80)
    baseline, intervention = twin.test_intervention('quit_smoking', years=10)
    
    print("\nDiabetes risk:")
    print(f"  Baseline: {baseline.disease_risks['diabetes']*100:.1f}%")
    print(f"  After quitting: {intervention.disease_risks['diabetes']*100:.1f}%")
    print(f"  Reduction: {(baseline.disease_risks['diabetes'] - intervention.disease_risks['diabetes'])*100:.1f}%")
    
    print("\n✅ Digital Twin Simulator Ready!")
    print("   - Natural language input parsing")
    print("   - 10-year trajectory simulation")
    print("   - Disease risk prediction")
    print("   - Intervention testing")
