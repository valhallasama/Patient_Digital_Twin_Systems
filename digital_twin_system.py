"""
Complete Digital Twin System - End-to-End Integration
Connects all modules: LLM parsing, timeline simulation, Markov models, 
multi-agent reasoning, ML prediction, and knowledge graph
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json

# Import all system components
from ai_core.llm_medical_parser import LLMMedicalParser, StructuredPatientData
from ai_core.llm_agent_base import MultiAgentLLMSystem, CardiologyLLMAgent, EndocrinologyLLMAgent
from agents.agent_communication import SwarmReasoningCoordinator
from agents.cardiology_agent import CardiologyAgent
from agents.metabolic_agent import MetabolicAgent
from agents.lifestyle_agent import LifestyleAgent
from simulation_engine.patient_timeline import PatientTimelineEngine, HealthState, PatientTimeline
from simulation_engine.markov_disease_model import (
    DiabetesMarkovModel, CVDMarkovModel, CKDMarkovModel, 
    MultiDiseaseMarkovSimulator, DiabetesState, CVDState, CKDState
)
from prediction_engine.temporal_models import SurvivalAnalysisModel, TemporalRiskPredictor
from prediction_engine.risk_predictor import RiskPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientDigitalTwin:
    """
    Complete Patient Digital Twin
    Integrates all system components for comprehensive health modeling
    """
    
    def __init__(self, patient_id: str, use_llm: bool = False, llm_api_key: Optional[str] = None):
        self.patient_id = patient_id
        self.use_llm = use_llm
        
        # Initialize all components
        logger.info(f"\n{'='*80}")
        logger.info(f"Initializing Digital Twin System for Patient: {patient_id}")
        logger.info(f"{'='*80}")
        
        # 1. Medical Report Parser (LLM-based)
        if use_llm:
            self.medical_parser = LLMMedicalParser(
                model_provider="openai",
                model_name="gpt-4",
                api_key=llm_api_key
            )
            logger.info("✓ LLM Medical Parser initialized")
        else:
            self.medical_parser = LLMMedicalParser(model_provider="openai", api_key=None)
            logger.info("✓ Rule-based Medical Parser initialized (fallback)")
        
        # 2. Patient Timeline Engine
        self.timeline_engine = PatientTimelineEngine()
        logger.info("✓ Patient Timeline Engine initialized")
        
        # 3. Markov Disease Models
        self.diabetes_markov = DiabetesMarkovModel()
        self.cvd_markov = CVDMarkovModel()
        self.ckd_markov = CKDMarkovModel()
        self.multi_disease_simulator = MultiDiseaseMarkovSimulator()
        logger.info("✓ Markov Disease Models initialized")
        
        # 4. Multi-Agent System (Rule-based)
        self.swarm_coordinator = SwarmReasoningCoordinator()
        self.swarm_coordinator.register_agent('Cardiology', CardiologyAgent())
        self.swarm_coordinator.register_agent('Metabolic', MetabolicAgent())
        self.swarm_coordinator.register_agent('Lifestyle', LifestyleAgent())
        logger.info("✓ Multi-Agent Swarm System initialized")
        
        # 5. LLM-Powered Agents (if available)
        if use_llm:
            self.llm_agent_system = MultiAgentLLMSystem(
                model_provider="openai",
                model_name="gpt-4",
                api_key=llm_api_key
            )
            self.llm_agent_system.register_agent(
                CardiologyLLMAgent(model_provider="openai", api_key=llm_api_key)
            )
            self.llm_agent_system.register_agent(
                EndocrinologyLLMAgent(model_provider="openai", api_key=llm_api_key)
            )
            logger.info("✓ LLM-Powered Agent System initialized")
        else:
            self.llm_agent_system = None
            logger.info("○ LLM-Powered Agents not available (no API key)")
        
        # 6. ML Prediction Models
        self.risk_predictor = RiskPredictor()
        logger.info("✓ ML Risk Predictor initialized")
        
        # 7. Temporal Models
        self.survival_model = SurvivalAnalysisModel()
        self.temporal_predictor = TemporalRiskPredictor()
        logger.info("✓ Temporal Models initialized")
        
        # Patient data storage
        self.patient_timeline: Optional[PatientTimeline] = None
        self.current_state: Optional[HealthState] = None
        self.structured_data: Optional[StructuredPatientData] = None
        self.analysis_results: Dict[str, Any] = {}
        
        logger.info(f"\n{'='*80}")
        logger.info("Digital Twin System Ready")
        logger.info(f"{'='*80}\n")
    
    def ingest_medical_report(self, medical_report_text: str) -> StructuredPatientData:
        """
        Step 1: Parse unstructured medical report into structured data
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: MEDICAL REPORT INGESTION")
        logger.info("="*80)
        
        self.structured_data = self.medical_parser.parse(medical_report_text)
        
        logger.info("✓ Medical report parsed successfully")
        logger.info(f"  Age: {self.structured_data.demographics.age}")
        logger.info(f"  BMI: {self.structured_data.physical.bmi}")
        logger.info(f"  BP: {self.structured_data.vitals.systolic_bp}/{self.structured_data.vitals.diastolic_bp}")
        
        return self.structured_data
    
    def initialize_patient_timeline(self, birth_date: datetime, 
                                   gender: str, ethnicity: str) -> PatientTimeline:
        """
        Step 2: Create patient timeline from structured data
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: PATIENT TIMELINE INITIALIZATION")
        logger.info("="*80)
        
        if not self.structured_data:
            raise ValueError("Must ingest medical report first")
        
        # Create initial health state from structured data
        initial_state = HealthState(
            timestamp=datetime.now(),
            weight_kg=self.structured_data.physical.weight_kg or 70,
            bmi=self.structured_data.physical.bmi or 25,
            waist_circumference_cm=self.structured_data.physical.waist_circumference_cm or 85,
            systolic_bp=self.structured_data.vitals.systolic_bp or 120,
            diastolic_bp=self.structured_data.vitals.diastolic_bp or 80,
            heart_rate=self.structured_data.vitals.heart_rate or 70,
            glucose_mmol_l=self.structured_data.labs.glucose_mmol_l or 5.0,
            hba1c_percent=self.structured_data.labs.hba1c_percent or 5.5,
            insulin_resistance_index=1.0,
            beta_cell_function_percent=100.0,
            total_cholesterol_mmol_l=self.structured_data.labs.total_cholesterol_mmol_l or 5.0,
            ldl_cholesterol_mmol_l=self.structured_data.labs.ldl_cholesterol_mmol_l or 3.0,
            hdl_cholesterol_mmol_l=self.structured_data.labs.hdl_cholesterol_mmol_l or 1.5,
            triglycerides_mmol_l=self.structured_data.labs.triglycerides_mmol_l or 1.5,
            egfr=self.structured_data.labs.egfr or 90,
            liver_function_index=1.0,
            exercise_hours_per_week=self.structured_data.lifestyle.exercise_hours_per_week or 2.0,
            smoking_status=self.structured_data.lifestyle.smoking_status or 'never',
            alcohol_units_per_week=self.structured_data.lifestyle.alcohol_units_per_week or 0,
            sleep_hours_per_night=self.structured_data.lifestyle.sleep_hours_per_night or 7.0,
            stress_level=0.5
        )
        
        self.current_state = initial_state
        
        # Create timeline
        self.patient_timeline = self.timeline_engine.create_timeline(
            patient_id=self.patient_id,
            initial_state=initial_state,
            birth_date=birth_date,
            gender=gender,
            ethnicity=ethnicity
        )
        
        logger.info("✓ Patient timeline initialized")
        logger.info(f"  Initial state recorded at {initial_state.timestamp}")
        
        return self.patient_timeline
    
    def run_multi_agent_analysis(self) -> Dict[str, Any]:
        """
        Step 3: Multi-agent collaborative analysis
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MULTI-AGENT COLLABORATIVE ANALYSIS")
        logger.info("="*80)
        
        # Prepare patient data for agents
        patient_data = self._prepare_patient_data_for_agents()
        
        # Run swarm reasoning (rule-based agents)
        swarm_results = self.swarm_coordinator.collaborative_diagnosis(patient_data)
        
        # Run LLM agents if available
        llm_results = None
        if self.llm_agent_system:
            logger.info("\nRunning LLM-powered agent analysis...")
            llm_results = self.llm_agent_system.analyze_patient(patient_data)
        
        self.analysis_results['multi_agent'] = {
            'swarm_reasoning': swarm_results,
            'llm_agents': llm_results
        }
        
        logger.info("✓ Multi-agent analysis complete")
        
        return self.analysis_results['multi_agent']
    
    def simulate_disease_progression(self, years: int = 10) -> Dict[str, Any]:
        """
        Step 4: Simulate disease progression using multiple models
        """
        logger.info("\n" + "="*80)
        logger.info(f"STEP 4: DISEASE PROGRESSION SIMULATION ({years} years)")
        logger.info("="*80)
        
        results = {}
        
        # 1. Timeline-based mechanistic simulation
        logger.info("\n1. Mechanistic Timeline Simulation")
        logger.info("-" * 80)
        
        timeline = self.timeline_engine.simulate_forward(
            self.patient_id,
            years=years,
            timestep_months=6
        )
        
        results['timeline_simulation'] = {
            'num_states': len(timeline.states),
            'events': timeline.events,
            'final_state': timeline.states[-1]
        }
        
        logger.info(f"✓ Simulated {len(timeline.states)} time points")
        logger.info(f"✓ Detected {len(timeline.events)} major events")
        
        # 2. Markov disease progression
        logger.info("\n2. Markov Disease Progression")
        logger.info("-" * 80)
        
        patient_factors = self._extract_patient_factors()
        
        initial_states = {
            'diabetes': DiabetesState.HEALTHY.value,
            'cvd': CVDState.HEALTHY.value,
            'ckd': CKDState.NORMAL.value
        }
        
        markov_trajectory = self.multi_disease_simulator.simulate_patient(
            initial_states=initial_states,
            patient_factors=patient_factors,
            years=years
        )
        
        results['markov_progression'] = markov_trajectory
        
        logger.info(f"✓ Markov simulation complete")
        logger.info(f"  Final diabetes state: {DiabetesState(markov_trajectory.iloc[-1]['diabetes_state']).name}")
        logger.info(f"  Final CVD state: {CVDState(markov_trajectory.iloc[-1]['cvd_state']).name}")
        
        self.analysis_results['disease_progression'] = results
        
        return results
    
    def predict_risks(self) -> Dict[str, Any]:
        """
        Step 5: ML-based risk prediction
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: ML RISK PREDICTION")
        logger.info("="*80)
        
        patient_data = self._prepare_patient_data_for_agents()
        
        # Traditional risk prediction
        risks = self.risk_predictor.predict_all_risks(patient_data)
        
        self.analysis_results['ml_predictions'] = risks
        
        logger.info("✓ Risk predictions complete")
        for disease, risk in risks.items():
            logger.info(f"  {disease}: {risk:.1%}")
        
        return risks
    
    def simulate_intervention(self, intervention_type: str, 
                            parameters: Dict, years: int = 10) -> Dict[str, Any]:
        """
        Step 6: Simulate intervention effects
        """
        logger.info("\n" + "="*80)
        logger.info(f"STEP 6: INTERVENTION SIMULATION - {intervention_type}")
        logger.info("="*80)
        
        # Apply intervention to timeline
        self.timeline_engine.apply_intervention(
            patient_id=self.patient_id,
            intervention_type=intervention_type,
            parameters=parameters,
            start_time=datetime.now()
        )
        
        # Simulate with intervention
        timeline_with_intervention = self.timeline_engine.simulate_forward(
            self.patient_id,
            years=years,
            timestep_months=6
        )
        
        logger.info(f"✓ Intervention simulation complete")
        logger.info(f"  Events with intervention: {len(timeline_with_intervention.events)}")
        
        return {
            'intervention': intervention_type,
            'parameters': parameters,
            'timeline': timeline_with_intervention
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive digital twin report
        """
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE DIGITAL TWIN REPORT")
        logger.info("="*80)
        
        report = {
            'patient_id': self.patient_id,
            'generated_at': datetime.now().isoformat(),
            'structured_data': self.medical_parser.to_dict(self.structured_data) if self.structured_data else None,
            'current_state': {
                'bmi': self.current_state.bmi if self.current_state else None,
                'bp': f"{self.current_state.systolic_bp}/{self.current_state.diastolic_bp}" if self.current_state else None,
                'hba1c': self.current_state.hba1c_percent if self.current_state else None,
            },
            'multi_agent_analysis': self.analysis_results.get('multi_agent'),
            'disease_progression': self.analysis_results.get('disease_progression'),
            'ml_predictions': self.analysis_results.get('ml_predictions'),
            'timeline_summary': {
                'num_states': len(self.patient_timeline.states) if self.patient_timeline else 0,
                'num_events': len(self.patient_timeline.events) if self.patient_timeline else 0,
            }
        }
        
        logger.info("✓ Comprehensive report generated")
        
        return report
    
    def _prepare_patient_data_for_agents(self) -> Dict:
        """Convert structured data to format expected by agents"""
        if not self.structured_data:
            return {}
        
        return {
            'demographics': {
                'age': self.structured_data.demographics.age,
                'gender': self.structured_data.demographics.gender,
            },
            'physical': {
                'bmi': self.structured_data.physical.bmi,
                'waist_circumference_cm': self.structured_data.physical.waist_circumference_cm,
            },
            'vitals': {
                'systolic_bp': self.structured_data.vitals.systolic_bp,
                'diastolic_bp': self.structured_data.vitals.diastolic_bp,
                'heart_rate': self.structured_data.vitals.heart_rate,
            },
            'labs': {
                'glucose_mmol_l': self.structured_data.labs.glucose_mmol_l,
                'hba1c_percent': self.structured_data.labs.hba1c_percent,
                'ldl_cholesterol_mmol_l': self.structured_data.labs.ldl_cholesterol_mmol_l,
                'hdl_cholesterol_mmol_l': self.structured_data.labs.hdl_cholesterol_mmol_l,
                'triglycerides_mmol_l': self.structured_data.labs.triglycerides_mmol_l,
            },
            'lifestyle': {
                'smoking_status': self.structured_data.lifestyle.smoking_status,
                'exercise_hours_per_week': self.structured_data.lifestyle.exercise_hours_per_week,
                'alcohol_units_per_week': self.structured_data.lifestyle.alcohol_units_per_week,
            },
            'family_history': {
                'father_cvd': self.structured_data.family_history.father_cvd,
                'father_cvd_age': self.structured_data.family_history.father_cvd_age,
            }
        }
    
    def _extract_patient_factors(self) -> Dict:
        """Extract patient factors for Markov models"""
        if not self.current_state:
            return {}
        
        age = (datetime.now() - self.patient_timeline.birth_date).days / 365.25 if self.patient_timeline else 50
        
        return {
            'age': age,
            'bmi': self.current_state.bmi,
            'hba1c': self.current_state.hba1c_percent,
            'systolic_bp': self.current_state.systolic_bp,
            'ldl': self.current_state.ldl_cholesterol_mmol_l,
            'smoking': self.current_state.smoking_status == 'current',
            'exercise_hours': self.current_state.exercise_hours_per_week,
            'on_metformin': self.current_state.on_metformin,
            'on_statin': self.current_state.on_statin,
            'has_diabetes': self.current_state.has_diabetes,
            'has_hypertension': self.current_state.has_hypertension,
        }


# Example usage - Complete end-to-end workflow
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PATIENT DIGITAL TWIN SYSTEM - END-TO-END DEMONSTRATION")
    print("="*80)
    
    # Sample medical report
    medical_report = """
    Patient: 55-year-old male, office worker
    
    Physical Examination:
    - Height: 175 cm, Weight: 95 kg, BMI: 31
    - Waist circumference: 105 cm
    - Blood pressure: 145/92 mmHg
    - Heart rate: 78 bpm
    
    Laboratory Results:
    - Fasting glucose: 6.2 mmol/L
    - HbA1c: 6.0%
    - Total cholesterol: 5.8 mmol/L
    - LDL cholesterol: 4.2 mmol/L
    - HDL cholesterol: 1.0 mmol/L
    - Triglycerides: 2.5 mmol/L
    - eGFR: 85 mL/min/1.73m²
    
    Social History:
    - Current smoker: 20 pack-years
    - Alcohol: 15 units per week
    - Exercise: 1 hour per week
    - Sleep: 5-6 hours per night
    - High stress job
    
    Family History:
    - Father: Myocardial infarction at age 58
    - Mother: Type 2 diabetes diagnosed at age 62
    
    Assessment:
    - Metabolic syndrome
    - Prediabetes
    - Hypertension (Stage 1)
    - High cardiovascular risk
    """
    
    # Initialize Digital Twin
    twin = PatientDigitalTwin(
        patient_id='DEMO_001',
        use_llm=False,  # Set to True with API key for full LLM features
        llm_api_key=None
    )
    
    # Step 1: Ingest medical report
    structured_data = twin.ingest_medical_report(medical_report)
    
    # Step 2: Initialize timeline
    timeline = twin.initialize_patient_timeline(
        birth_date=datetime(1969, 3, 15),
        gender='male',
        ethnicity='caucasian'
    )
    
    # Step 3: Multi-agent analysis
    agent_analysis = twin.run_multi_agent_analysis()
    
    # Step 4: Disease progression simulation
    progression = twin.simulate_disease_progression(years=10)
    
    # Step 5: Risk prediction
    risks = twin.predict_risks()
    
    # Step 6: Intervention simulation
    intervention = twin.simulate_intervention(
        intervention_type='lifestyle_modification',
        parameters={
            'exercise': 5.0,  # hours per week
            'stress_level': 0.3
        },
        years=10
    )
    
    # Generate comprehensive report
    report = twin.generate_comprehensive_report()
    
    # Display summary
    print("\n" + "="*80)
    print("DIGITAL TWIN ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nPatient: {twin.patient_id}")
    print(f"Current BMI: {twin.current_state.bmi:.1f}")
    print(f"Current BP: {twin.current_state.systolic_bp:.0f}/{twin.current_state.diastolic_bp:.0f}")
    print(f"Current HbA1c: {twin.current_state.hba1c_percent:.1f}%")
    
    print(f"\nTimeline:")
    print(f"  States recorded: {len(twin.patient_timeline.states)}")
    print(f"  Events detected: {len(twin.patient_timeline.events)}")
    
    if twin.patient_timeline.events:
        print(f"\nMajor Events:")
        for event in twin.patient_timeline.events:
            print(f"  • {event['description']}")
    
    print("\n" + "="*80)
    print("✓ DIGITAL TWIN SYSTEM DEMONSTRATION COMPLETE")
    print("="*80)
