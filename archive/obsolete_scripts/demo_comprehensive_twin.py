#!/usr/bin/env python3
"""
Comprehensive Digital Twin Demo
Demonstrates MiroFish-style medical simulation with medical theory-based imputation
"""

import json
from mirofish_engine.digital_twin_simulator import DigitalTwinSimulator


def demo_healthy_patient():
    """Demo: Healthy 30-year-old"""
    print("=" * 80)
    print("DEMO 1: HEALTHY 30-YEAR-OLD")
    print("=" * 80)
    
    # Minimal data - most will be imputed
    patient_data = {
        'patient_id': 'HEALTHY_001',
        'age': 30,
        'sex': 'M',
        'height': 175,  # cm
        'weight': 70,   # kg
        'lifestyle': {
            'physical_activity': 'vigorous',
            'diet_quality': 'good',
            'smoking_status': 'never',
            'alcohol_consumption': 'light',
            'sleep_duration': 8,
            'stress_level': 'low'
        }
    }
    
    print("\n📋 Input Data:")
    print(json.dumps(patient_data, indent=2))
    
    # Create and run simulation
    simulator = DigitalTwinSimulator(patient_data)
    results = simulator.simulate(years=10, timestep='month')
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Current Health Score: {results['current_state']['overall_health_score']:.1f}/10")
    
    print("\n🏥 Organ Health:")
    for organ, score in results['current_state']['organ_health'].items():
        print(f"   {organ.capitalize():15} {score:.1f}/10")
    
    print("\n⚠️  Disease Predictions (10-year risk):")
    for i, pred in enumerate(results['disease_predictions'][:5], 1):
        print(f"   {i}. {pred['disease']:30} {pred['probability']*100:5.1f}% "
              f"(~{pred['time_to_onset_years']:.0f} years)")
    
    print(f"\n📈 Data Completeness: {results['metadata']['data_completeness']*100:.0f}%")
    print("   (Missing data was imputed using medical theory)")
    
    # Save results
    simulator.save_results()
    
    return results


def demo_prediabetic_patient():
    """Demo: 45-year-old with prediabetes"""
    print("\n\n" + "=" * 80)
    print("DEMO 2: 45-YEAR-OLD WITH PREDIABETES")
    print("=" * 80)
    
    patient_data = {
        'patient_id': 'PREDIABETIC_001',
        'age': 45,
        'sex': 'F',
        'height': 165,
        'weight': 80,
        'fasting_glucose': 110,  # Prediabetic range
        'hba1c': 5.9,           # Prediabetic
        'blood_pressure': {
            'systolic': 135,
            'diastolic': 85
        },
        'total_cholesterol': 220,
        'ldl_cholesterol': 140,
        'hdl_cholesterol': 45,
        'family_history': {
            'diabetes': True,
            'cardiovascular_disease': True
        },
        'lifestyle': {
            'physical_activity': 'light',
            'diet_quality': 'fair',
            'smoking_status': 'never',
            'alcohol_consumption': 'none',
            'sleep_duration': 6,
            'stress_level': 'high'
        }
    }
    
    print("\n📋 Input Data:")
    print(json.dumps(patient_data, indent=2))
    
    simulator = DigitalTwinSimulator(patient_data)
    results = simulator.simulate(years=5, timestep='month')
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Current Health Score: {results['current_state']['overall_health_score']:.1f}/10")
    
    print("\n🏥 Organ Health:")
    for organ, score in results['current_state']['organ_health'].items():
        print(f"   {organ.capitalize():15} {score:.1f}/10")
    
    print("\n⚠️  Disease Predictions (5-year risk):")
    for i, pred in enumerate(results['disease_predictions'][:5], 1):
        risk_factors = pred.get('risk_factors', [])
        print(f"\n   {i}. {pred['disease']}")
        print(f"      Risk: {pred['probability']*100:.1f}% in ~{pred['time_to_onset_years']:.0f} years")
        if risk_factors:
            print(f"      Factors: {', '.join(risk_factors[:3])}")
    
    print("\n💊 Intervention Recommendations:")
    for intervention_set in results['interventions'][:2]:
        print(f"\n   For {intervention_set['disease']}:")
        print(f"   Current Risk: {intervention_set['current_risk']*100:.1f}%")
        
        for rec in intervention_set['recommendations'][:3]:
            print(f"\n      • {rec['intervention']}")
            print(f"        Risk Reduction: {rec['risk_reduction']*100:.0f}% → "
                  f"New Risk: {rec['new_probability']*100:.1f}%")
            print(f"        Evidence: {rec['evidence']}")
            print(f"        Difficulty: {rec['difficulty']}")
    
    simulator.save_results()
    
    return results


def demo_minimal_data_patient():
    """Demo: Patient with very minimal data - heavy imputation"""
    print("\n\n" + "=" * 80)
    print("DEMO 3: MINIMAL DATA (HEAVY IMPUTATION)")
    print("=" * 80)
    
    # Only age and sex provided
    patient_data = {
        'patient_id': 'MINIMAL_001',
        'age': 55,
        'sex': 'M'
    }
    
    print("\n📋 Input Data (MINIMAL):")
    print(json.dumps(patient_data, indent=2))
    print("\n⚠️  Most parameters will be imputed using medical theory!")
    
    simulator = DigitalTwinSimulator(patient_data)
    results = simulator.simulate(years=5, timestep='month')
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Current Health Score: {results['current_state']['overall_health_score']:.1f}/10")
    print(f"📈 Data Completeness: {results['metadata']['data_completeness']*100:.0f}%")
    print("   (All missing data imputed from age/sex)")
    
    print("\n⚠️  Disease Predictions:")
    for i, pred in enumerate(results['disease_predictions'][:5], 1):
        print(f"   {i}. {pred['disease']:30} {pred['probability']*100:5.1f}% "
              f"(confidence: {pred.get('confidence', 0.5)*100:.0f}%)")
    
    print("\n⚠️  Note: Predictions based on imputed data have lower confidence")
    print("   Provide more lab values for better accuracy!")
    
    simulator.save_results()
    
    return results


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DIGITAL TWIN SYSTEM")
    print("MiroFish-Style Medical Simulation with Medical Theory Imputation")
    print("=" * 80)
    
    # Demo 1: Healthy patient
    demo_healthy_patient()
    
    # Demo 2: Prediabetic patient
    demo_prediabetic_patient()
    
    # Demo 3: Minimal data
    demo_minimal_data_patient()
    
    print("\n\n" + "=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print("\n✅ System demonstrates:")
    print("   • Medical theory-based data imputation")
    print("   • Multi-year trajectory simulation")
    print("   • Disease prediction with timing")
    print("   • Evidence-based intervention recommendations")
    print("   • Works with minimal or comprehensive data")
    print("\n📁 Results saved to: outputs/simulations/")


if __name__ == "__main__":
    main()
