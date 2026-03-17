#!/usr/bin/env python3
"""
Test NHANES Loader
Verify that NHANES data is downloaded and can be loaded correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_integration.nhanes_loader import NHANESLoader
from data_integration.data_harmonizer import DataHarmonizer
from data_integration.feature_extractor import FeatureExtractor


def test_nhanes_loader():
    """Test NHANES data loading"""
    
    print("=" * 80)
    print("TESTING NHANES DATA LOADER")
    print("=" * 80)
    
    # Initialize loader
    nhanes_path = './data/nhanes'
    print(f"\n1. Initializing NHANES loader...")
    print(f"   Path: {nhanes_path}")
    
    loader = NHANESLoader(nhanes_path, cycle='2017-2018')
    
    # Check if data exists
    if not loader.cycle_path.exists():
        print(f"\n❌ ERROR: NHANES data not found at {loader.cycle_path}")
        print(f"\nPlease run: ./download_nhanes.sh")
        return False
    
    print(f"   ✅ Data directory found")
    
    # Load demographics
    print(f"\n2. Loading demographics...")
    demo = loader.load_demographics()
    
    if demo.empty:
        print(f"   ❌ ERROR: Could not load demographics")
        print(f"   Make sure DEMO_J.XPT exists in {loader.cycle_path}")
        return False
    
    print(f"   ✅ Loaded {len(demo)} participants")
    print(f"   Age range: {demo['RIDAGEYR'].min():.0f} - {demo['RIDAGEYR'].max():.0f}")
    
    # Get a cohort
    print(f"\n3. Getting cohort (age 40-70, with labs)...")
    cohort = loader.get_cohort(min_age=40, max_age=70, has_labs=True, limit=100)
    
    if not cohort:
        print(f"   ❌ ERROR: No patients found in cohort")
        return False
    
    print(f"   ✅ Found {len(cohort)} patients")
    
    # Extract features for first 5 patients
    print(f"\n4. Extracting features for sample patients...")
    
    harmonizer = DataHarmonizer()
    extractor = FeatureExtractor()
    
    successful = 0
    for i, seqn in enumerate(cohort[:5]):
        try:
            # Extract raw features
            features = loader.extract_patient_features(seqn)
            
            # Harmonize
            harmonized = harmonizer.harmonize(features, source='nhanes')
            
            # Validate
            is_valid, errors = harmonizer.validate(harmonized)
            
            if is_valid:
                # Extract ML features
                ml_features = extractor.extract_all_features(harmonized)
                
                print(f"\n   Patient {i+1} (SEQN: {seqn}):")
                print(f"     Age: {harmonized.get('age')} years")
                print(f"     Sex: {harmonized.get('sex')}")
                print(f"     BMI: {harmonized.get('bmi'):.1f}" if harmonized.get('bmi') else "     BMI: N/A")
                print(f"     Glucose: {harmonized.get('fasting_glucose'):.1f} mg/dL" if harmonized.get('fasting_glucose') else "     Glucose: N/A")
                print(f"     HbA1c: {harmonized.get('hba1c'):.2f}%" if harmonized.get('hba1c') else "     HbA1c: N/A")
                print(f"     BP: {harmonized.get('systolic_bp'):.0f}/{harmonized.get('diastolic_bp'):.0f}" if harmonized.get('systolic_bp') else "     BP: N/A")
                print(f"     Activity: {harmonized.get('physical_activity')}")
                print(f"     ML Features: {len(ml_features)} features extracted")
                
                successful += 1
            else:
                print(f"\n   Patient {i+1}: ❌ Validation failed: {errors}")
        
        except Exception as e:
            print(f"\n   Patient {i+1}: ❌ Error: {e}")
    
    print(f"\n5. Summary:")
    print(f"   ✅ Successfully processed {successful}/5 patients")
    
    if successful >= 3:
        print(f"\n" + "=" * 80)
        print(f"✅ NHANES LOADER TEST PASSED")
        print(f"=" * 80)
        print(f"\nNext steps:")
        print(f"1. Process full cohort: python3 examples/process_nhanes_data.py")
        print(f"2. Train model: python3 train_hybrid_model.py --data nhanes")
        return True
    else:
        print(f"\n" + "=" * 80)
        print(f"⚠️  NHANES LOADER TEST PARTIALLY FAILED")
        print(f"=" * 80)
        print(f"\nSome patients could not be processed. This is normal if data is incomplete.")
        return False


if __name__ == '__main__':
    success = test_nhanes_loader()
    sys.exit(0 if success else 1)
