"""
Test Real NHANES Data Integration

Tests the updated NHANESCSVLoader with variable mapping on real NHANES dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_integration.nhanes_csv_loader import NHANESCSVLoader
from data_integration.nhanes_variable_mapping import NHANESVariableMapper
import pandas as pd
import numpy as np


def test_data_loading():
    """Test basic data loading"""
    print("=" * 80)
    print("TEST 1: Data Loading")
    print("=" * 80)
    
    loader = NHANESCSVLoader(data_path="./data/nhanes/raw_csv")
    
    print("\n1. Loading demographics...")
    demo = loader.load_demographics()
    print(f"   ✓ Loaded {len(demo):,} patients")
    print(f"   ✓ Columns: {len(demo.columns)}")
    print(f"   ✓ Sample columns: {list(demo.columns[:10])}")
    
    print("\n2. Loading questionnaire...")
    quest = loader.load_questionnaire()
    print(f"   ✓ Loaded {len(quest):,} questionnaire records")
    print(f"   ✓ Columns: {len(quest.columns)}")
    
    print("\n3. Loading chemicals (lab data)...")
    chem = loader.load_chemicals()
    print(f"   ✓ Loaded {len(chem):,} chemical measurements")
    print(f"   ✓ Columns: {len(chem.columns)}")
    
    print("\n4. Loading medications...")
    meds = loader.load_medications()
    print(f"   ✓ Loaded {len(meds):,} medication records")
    
    return loader


def test_variable_mapping():
    """Test variable mapping system"""
    print("\n" + "=" * 80)
    print("TEST 2: Variable Mapping")
    print("=" * 80)
    
    mapper = NHANESVariableMapper()
    
    print(f"\n✓ Demographics mappings: {len(mapper.demographics_map)}")
    print("   Sample mappings:")
    for csv_var, std_var in list(mapper.demographics_map.items())[:5]:
        print(f"     {csv_var:15} → {std_var}")
    
    print(f"\n✓ Lab/chemical mappings: {len(mapper.lab_map)}")
    print("   Sample mappings:")
    for csv_var, std_var in list(mapper.lab_map.items())[:10]:
        print(f"     {csv_var:15} → {std_var}")
    
    print(f"\n✓ Questionnaire mappings: {len(mapper.questionnaire_map)}")
    print("   Sample mappings:")
    for csv_var, std_var in list(mapper.questionnaire_map.items())[:10]:
        print(f"     {csv_var:15} → {std_var}")
    
    print(f"\n✓ Derived calculations: {len(mapper.derived_calculations)}")
    for calc_name in mapper.derived_calculations.keys():
        print(f"     - {calc_name}")
    
    return mapper


def test_patient_extraction(loader):
    """Test extracting individual patient data"""
    print("\n" + "=" * 80)
    print("TEST 3: Patient Data Extraction")
    print("=" * 80)
    
    demo = loader.load_demographics()
    sample_seqns = demo['SEQN'].head(10).tolist()
    
    print(f"\nTesting extraction for {len(sample_seqns)} sample patients...\n")
    
    successful = 0
    for i, seqn in enumerate(sample_seqns, 1):
        print(f"{i}. Patient SEQN={seqn}")
        
        features = loader.extract_patient_features(seqn)
        
        if features:
            successful += 1
            print(f"   ✓ Extracted {len(features)} features")
            
            key_features = ['age', 'sex', 'bmi', 'systolic_bp', 'glucose', 'hba1c']
            available = [f for f in key_features if f in features and features[f] is not None]
            print(f"   ✓ Key features available: {available}")
            
            if 'age' in features:
                print(f"     - Age: {features['age']}")
            if 'sex' in features:
                print(f"     - Sex: {features['sex']}")
            if 'bmi' in features:
                print(f"     - BMI: {features['bmi']:.1f}")
            if 'systolic_bp' in features:
                print(f"     - BP: {features['systolic_bp']:.0f} mmHg")
            if 'hba1c' in features:
                print(f"     - HbA1c: {features['hba1c']:.1f}%")
            if 'egfr' in features and features['egfr']:
                print(f"     - eGFR: {features['egfr']:.1f} mL/min/1.73m²")
        else:
            print(f"   ✗ No features extracted")
        
        print()
    
    print(f"Success rate: {successful}/{len(sample_seqns)} ({100*successful/len(sample_seqns):.1f}%)")
    
    return successful > 0


def test_cohort_extraction(loader):
    """Test extracting a patient cohort"""
    print("\n" + "=" * 80)
    print("TEST 4: Cohort Extraction")
    print("=" * 80)
    
    print("\nExtracting cohort of 100 patients (ages 18-90)...")
    
    cohort = loader.get_patient_cohort(max_patients=100, min_age=18, max_age=90)
    
    print(f"\n✓ Extracted {len(cohort)} patients with complete data")
    
    if cohort:
        ages = [p['age'] for p in cohort if 'age' in p]
        sexes = [p['sex'] for p in cohort if 'sex' in p]
        bmis = [p['bmi'] for p in cohort if 'bmi' in p and p['bmi'] is not None]
        
        print(f"\nCohort Statistics:")
        print(f"  Age: {np.mean(ages):.1f} ± {np.std(ages):.1f} years (range: {min(ages):.0f}-{max(ages):.0f})")
        print(f"  Sex: {sexes.count('male')} male, {sexes.count('female')} female")
        if bmis:
            print(f"  BMI: {np.mean(bmis):.1f} ± {np.std(bmis):.1f} kg/m² (n={len(bmis)})")
        
        feature_counts = {}
        for patient in cohort:
            for feature in patient.keys():
                if patient[feature] is not None:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        print(f"\nFeature Availability (top 15):")
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features[:15]:
            pct = 100 * count / len(cohort)
            print(f"  {feature:25} {count:4}/{len(cohort)} ({pct:5.1f}%)")
    
    return len(cohort) > 0


def test_disease_labels(loader):
    """Test disease label extraction"""
    print("\n" + "=" * 80)
    print("TEST 5: Disease Labels")
    print("=" * 80)
    
    print("\nExtracting 1000 patients to check disease prevalence...")
    
    cohort = loader.get_patient_cohort(max_patients=1000, min_age=18, max_age=90)
    
    if not cohort:
        print("✗ No cohort extracted")
        return False
    
    diabetes_count = 0
    hypertension_count = 0
    
    for patient in cohort:
        if patient.get('hba1c') and patient['hba1c'] >= 6.5:
            diabetes_count += 1
        elif patient.get('doctor_told_diabetes'):
            diabetes_count += 1
        
        if patient.get('systolic_bp') and patient['systolic_bp'] >= 140:
            hypertension_count += 1
        elif patient.get('doctor_told_hypertension'):
            hypertension_count += 1
    
    print(f"\nDisease Prevalence in Cohort (n={len(cohort)}):")
    print(f"  Diabetes:     {diabetes_count:4} ({100*diabetes_count/len(cohort):5.1f}%)")
    print(f"  Hypertension: {hypertension_count:4} ({100*hypertension_count/len(cohort):5.1f}%)")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("NHANES REAL DATA INTEGRATION TEST")
    print("=" * 80)
    print("\nTesting updated NHANESCSVLoader with variable mapping system")
    print("Dataset: NHANES 1988-2018 Harmonized CSV")
    print()
    
    try:
        loader = test_data_loading()
        
        mapper = test_variable_mapping()
        
        patient_success = test_patient_extraction(loader)
        
        cohort_success = test_cohort_extraction(loader)
        
        disease_success = test_disease_labels(loader)
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"✓ Data Loading:          PASSED")
        print(f"✓ Variable Mapping:      PASSED")
        print(f"{'✓' if patient_success else '✗'} Patient Extraction:    {'PASSED' if patient_success else 'FAILED'}")
        print(f"{'✓' if cohort_success else '✗'} Cohort Extraction:     {'PASSED' if cohort_success else 'FAILED'}")
        print(f"{'✓' if disease_success else '✗'} Disease Labels:        {'PASSED' if disease_success else 'FAILED'}")
        
        if patient_success and cohort_success and disease_success:
            print("\n" + "=" * 80)
            print("✓ ALL TESTS PASSED - REAL NHANES DATA READY FOR TRAINING!")
            print("=" * 80)
            return True
        else:
            print("\n" + "=" * 80)
            print("✗ SOME TESTS FAILED - CHECK VARIABLE MAPPINGS")
            print("=" * 80)
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
