#!/usr/bin/env python3
"""
Comprehensive Test Suite for Temporal Simulation System
Tests parameter evolution, disease prediction, and lifestyle impact
"""

from mirofish_engine.digital_twin_simulator import DigitalTwinSimulator
import json
from datetime import datetime


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_scenario(name, patient_data, expected_behavior):
    """Test a single patient scenario"""
    print(f"\n📋 Test: {name}")
    print(f"Expected: {expected_behavior}")
    print("-" * 80)
    
    # Print patient info
    print(f"Patient ID: {patient_data['patient_id']}")
    print(f"Age: {patient_data['age']}")
    print(f"Starting HbA1c: {patient_data.get('hba1c', 'N/A')}%")
    
    lifestyle = patient_data.get('lifestyle', {})
    print(f"Lifestyle:")
    print(f"  - Physical Activity: {lifestyle.get('physical_activity', 'N/A')}")
    print(f"  - Diet Quality: {lifestyle.get('diet_quality', 'N/A')}")
    print(f"  - Smoking: {lifestyle.get('smoking_status', 'N/A')}")
    print(f"  - Stress: {lifestyle.get('stress_level', 'N/A')}")
    
    # Run simulation
    print(f"\n🔬 Running 2-year simulation...")
    sim = DigitalTwinSimulator(patient_data)
    results = sim.simulate(years=2, timestep='month')
    
    # Print results
    print(f"\n📊 Results:")
    print(f"Overall Health Score: {results['current_state']['overall_health_score']:.1f}/10")
    
    # Get diabetes prediction
    diabetes_pred = None
    for pred in results['disease_predictions']:
        if 'Diabetes' in pred['disease']:
            diabetes_pred = pred
            break
    
    if diabetes_pred:
        print(f"\n🎯 Diabetes Prediction:")
        print(f"  Status: {diabetes_pred.get('status', 'N/A')}")
        print(f"  Probability: {diabetes_pred['probability']*100:.1f}%")
        
        if 'time_to_onset_days' in diabetes_pred:
            days = diabetes_pred['time_to_onset_days']
            years = diabetes_pred['time_to_onset_years']
            if days > 0:
                print(f"  Time to Onset: {days} days ({years:.1f} years)")
            elif days == 0:
                print(f"  Time to Onset: Already present")
        
        if 'current_hba1c' in diabetes_pred:
            print(f"  Current HbA1c: {diabetes_pred['current_hba1c']}%")
        
        if 'projected_hba1c' in diabetes_pred:
            print(f"  Projected HbA1c (1 year): {diabetes_pred['projected_hba1c']}%")
        
        if 'progression_rate' in diabetes_pred:
            print(f"  Progression Rate: {diabetes_pred['progression_rate']}")
        
        if diabetes_pred.get('risk_factors'):
            print(f"  Risk Factors: {', '.join(diabetes_pred['risk_factors'])}")
    
    # Check trajectory
    if len(results['trajectory']) > 0:
        first_state = results['trajectory'][0]
        last_state = results['trajectory'][-1]
        
        # Check if agents data exists in trajectory
        if 'agents' in first_state and 'agents' in last_state:
            if 'metabolic' in first_state['agents'] and 'metabolic' in last_state['agents']:
                start_hba1c = first_state['agents']['metabolic'].get('hba1c', 0)
                end_hba1c = last_state['agents']['metabolic'].get('hba1c', 0)
                change = end_hba1c - start_hba1c
                
                print(f"\n📈 HbA1c Trajectory:")
                print(f"  Start: {start_hba1c:.2f}%")
                print(f"  End (2 years): {end_hba1c:.2f}%")
                print(f"  Change: {change:+.2f}%")
    
    print("\n✅ Test Complete")
    return results


def main():
    """Run all test scenarios"""
    print_section("TEMPORAL SIMULATION TEST SUITE")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Healthy patient with poor lifestyle
    print_section("TEST 1: Healthy Patient with Poor Lifestyle")
    test_scenario(
        "Healthy → Poor Lifestyle → Should Progress to Diabetes",
        {
            'patient_id': 'TEST_POOR_LIFESTYLE',
            'age': 35,
            'sex': 'M',
            'height': 175,
            'weight': 75,
            'hba1c': 5.0,
            'fasting_glucose': 90,
            'bmi': 24.5,
            'lifestyle': {
                'physical_activity': 'sedentary',
                'diet_quality': 'poor',
                'smoking_status': 'current',
                'stress_level': 'high'
            }
        },
        "HbA1c should increase ~1-1.5% per year, diabetes onset in ~1-2 years"
    )
    
    # Test 2: Healthy patient with good lifestyle
    print_section("TEST 2: Healthy Patient with Good Lifestyle")
    test_scenario(
        "Healthy → Good Lifestyle → Should Stay Healthy",
        {
            'patient_id': 'TEST_GOOD_LIFESTYLE',
            'age': 35,
            'sex': 'F',
            'height': 165,
            'weight': 60,
            'hba1c': 5.0,
            'fasting_glucose': 85,
            'bmi': 22.0,
            'lifestyle': {
                'physical_activity': 'vigorous',
                'diet_quality': 'excellent',
                'smoking_status': 'never',
                'stress_level': 'low'
            }
        },
        "HbA1c should stay stable or decrease slightly, low diabetes risk"
    )
    
    # Test 3: Prediabetic with poor lifestyle
    print_section("TEST 3: Prediabetic with Poor Lifestyle")
    test_scenario(
        "Prediabetic → Poor Lifestyle → Should Progress Quickly",
        {
            'patient_id': 'TEST_PREDIABETIC_POOR',
            'age': 45,
            'sex': 'M',
            'height': 175,
            'weight': 90,
            'hba1c': 5.9,
            'fasting_glucose': 110,
            'bmi': 29.4,
            'lifestyle': {
                'physical_activity': 'sedentary',
                'diet_quality': 'poor',
                'smoking_status': 'current',
                'stress_level': 'high'
            }
        },
        "HbA1c should cross 6.5% threshold quickly, <1 year to diabetes"
    )
    
    # Test 4: Prediabetic with lifestyle intervention
    print_section("TEST 4: Prediabetic with Lifestyle Intervention")
    test_scenario(
        "Prediabetic → Good Lifestyle → Should Improve",
        {
            'patient_id': 'TEST_PREDIABETIC_GOOD',
            'age': 45,
            'sex': 'F',
            'height': 165,
            'weight': 75,
            'hba1c': 5.9,
            'fasting_glucose': 110,
            'bmi': 27.5,
            'lifestyle': {
                'physical_activity': 'vigorous',
                'diet_quality': 'excellent',
                'smoking_status': 'never',
                'stress_level': 'low'
            }
        },
        "HbA1c should decrease or stabilize, diabetes risk reduced"
    )
    
    # Test 5: Already diabetic
    print_section("TEST 5: Already Diabetic Patient")
    test_scenario(
        "Diabetic → Should Show Current Diagnosis",
        {
            'patient_id': 'TEST_DIABETIC',
            'age': 55,
            'sex': 'M',
            'height': 175,
            'weight': 95,
            'hba1c': 7.5,
            'fasting_glucose': 160,
            'bmi': 31.0,
            'lifestyle': {
                'physical_activity': 'light',
                'diet_quality': 'fair',
                'smoking_status': 'former',
                'stress_level': 'moderate'
            }
        },
        "Should show CURRENT DIAGNOSIS, time_to_onset = 0"
    )
    
    # Test 6: Minimal data (test imputation)
    print_section("TEST 6: Minimal Data (Test Imputation)")
    test_scenario(
        "Minimal Data → Should Impute and Predict",
        {
            'patient_id': 'TEST_MINIMAL',
            'age': 40,
            'sex': 'M',
            'lifestyle': {
                'physical_activity': 'sedentary',
                'diet_quality': 'poor'
            }
        },
        "Should impute missing values and still make predictions"
    )
    
    print_section("ALL TESTS COMPLETE")
    print("\n✅ Test suite finished successfully!")
    print("\nKey Findings to Verify:")
    print("1. Poor lifestyle → HbA1c increases over time")
    print("2. Good lifestyle → HbA1c stable or decreases")
    print("3. Prediabetic + poor lifestyle → Quick progression to diabetes")
    print("4. Already diabetic → Shows CURRENT DIAGNOSIS")
    print("5. System can predict exact days to disease onset")
    print("6. Missing data is properly imputed")


if __name__ == "__main__":
    main()
