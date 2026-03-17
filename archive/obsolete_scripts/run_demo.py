#!/usr/bin/env python3
"""
Demo script for Health Digital Twin Prediction Platform
Demonstrates the complete workflow from data generation to prediction
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent))

from synthetic_data_generator.patient_population_generator import PatientPopulationGenerator
from synthetic_data_generator.disease_progression_generator import DiseaseProgressionGenerator
from agents.base_agent import MultiAgentSystem
from agents.cardiology_agent import CardiologyAgent
from agents.metabolic_agent import MetabolicAgent
from agents.lifestyle_agent import LifestyleAgent
from prediction_engine.risk_predictor import RiskPredictor
from simulation_engine.intervention_simulator import InterventionSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_data_generation():
    print_header("STEP 1: Synthetic Patient Data Generation")
    
    generator = PatientPopulationGenerator()
    data = generator.generate_complete_population(n=100, output_dir="data/synthetic")
    
    logger.info(f"Generated {len(data['complete'])} patients")
    logger.info(f"\nSample patient:\n{data['complete'].iloc[0]}")
    
    return data['complete']


def demo_multi_agent_evaluation(patient_data):
    print_header("STEP 2: Multi-Agent Medical Evaluation")
    
    system = MultiAgentSystem()
    system.register_agent(CardiologyAgent())
    system.register_agent(MetabolicAgent())
    system.register_agent(LifestyleAgent())
    
    sample_patient = patient_data.iloc[0]
    
    logger.info(f"Evaluating patient: {sample_patient['patient_id']}")
    logger.info(f"Age: {sample_patient['age']}, Gender: {sample_patient['gender']}, BMI: {sample_patient['bmi']:.1f}")
    
    evaluation = system.evaluate_patient(sample_patient)
    
    logger.info("\n--- Multi-Agent Consensus ---")
    logger.info(f"Consensus Risk Score: {evaluation['consensus']['consensus']:.2%}")
    logger.info(f"Confidence: {evaluation['consensus']['confidence']:.2%}")
    
    logger.info("\n--- Recommendations ---")
    for i, rec in enumerate(evaluation['recommendations'][:5], 1):
        logger.info(f"{i}. {rec}")
    
    return evaluation


def demo_risk_prediction(patient_data):
    print_header("STEP 3: Disease Risk Prediction")
    
    predictor = RiskPredictor()
    
    sample_patient = patient_data.iloc[0]
    
    predictions = predictor.predict_all_risks(sample_patient, time_horizon_years=10)
    
    logger.info(f"10-Year Risk Predictions for {predictions['patient_id']}:")
    logger.info(f"\nCardiovascular Disease: {predictions['individual_risks']['cvd']['risk_percentage']:.1f}%")
    logger.info(f"Type 2 Diabetes: {predictions['individual_risks']['diabetes']['risk_percentage']:.1f}%")
    logger.info(f"Cancer: {predictions['individual_risks']['cancer']['risk_percentage']:.1f}%")
    logger.info(f"\nOverall Risk Level: {predictions['overall_risk_level'].upper()}")
    
    return predictions


def demo_intervention_simulation(patient_data):
    print_header("STEP 4: Intervention Simulation & Ranking")
    
    simulator = InterventionSimulator()
    
    sample_patient = patient_data.iloc[0]
    
    ranked_interventions = simulator.rank_interventions(sample_patient)
    
    logger.info("Top 5 Recommended Interventions:")
    for i, intervention in enumerate(ranked_interventions[:5], 1):
        logger.info(f"\n{i}. {intervention['intervention'].replace('_', ' ').title()}")
        logger.info(f"   Benefit Score: {intervention['benefit_score']:.1f}")
        logger.info(f"   Life Expectancy Gain: {intervention['life_expectancy_gain']:.1f} years")
        logger.info(f"   Total Risk Reduction: {intervention['total_risk_reduction']:.1%}")
        logger.info(f"   Adherence Rate: {intervention['adherence_rate']:.0%}")
    
    return ranked_interventions


def demo_disease_progression(patient_data):
    print_header("STEP 5: Disease Progression Simulation")
    
    prog_gen = DiseaseProgressionGenerator()
    
    sample_patient = patient_data.iloc[0]
    
    trajectories = prog_gen.simulate_disease_trajectory(sample_patient, years=10)
    
    logger.info(f"10-Year Disease Trajectory for {sample_patient['patient_id']}:")
    logger.info(f"\n{trajectories[['year', 'age', 'diabetes', 'cardiovascular_disease', 'cancer', 'alive']]}")
    
    final_state = trajectories.iloc[-1]
    logger.info(f"\nFinal State (Year 10):")
    logger.info(f"  Diabetes: {'Yes' if final_state['diabetes'] else 'No'}")
    logger.info(f"  CVD: {'Yes' if final_state['cardiovascular_disease'] else 'No'}")
    logger.info(f"  Cancer: {'Yes' if final_state['cancer'] else 'No'}")
    logger.info(f"  Alive: {'Yes' if final_state['alive'] else 'No'}")
    
    return trajectories


def main():
    print_header("Health Digital Twin Prediction Platform - Demo")
    
    logger.info("Starting comprehensive demo workflow...")
    
    try:
        patients = demo_data_generation()
        
        evaluation = demo_multi_agent_evaluation(patients)
        
        predictions = demo_risk_prediction(patients)
        
        interventions = demo_intervention_simulation(patients)
        
        trajectories = demo_disease_progression(patients)
        
        print_header("DEMO COMPLETE")
        
        logger.info("✅ All components demonstrated successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start API server: python api/api_server.py")
        logger.info("2. Launch dashboard: streamlit run dashboard/health_dashboard.py")
        logger.info("3. Explore synthetic data in: data/synthetic/")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
