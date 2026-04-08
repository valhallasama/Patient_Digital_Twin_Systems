#!/usr/bin/env python3
"""
Process NHANES Data for GNN-Transformer Training

Simplified processing that works directly with NHANES CSV loader output.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_integration.nhanes_csv_loader import NHANESCSVLoader
from data_integration.feature_extractor import FeatureExtractor
from data_integration.comprehensive_disease_labels import ComprehensiveDiseaseLabeler
import numpy as np
import pickle
from tqdm import tqdm

def process_nhanes_data(num_patients: int = 10000):
    """Process NHANES data for training"""
    
    print("="*80)
    print("PROCESSING NHANES DATA FOR GNN-TRANSFORMER TRAINING")
    print("="*80)
    
    # Initialize
    loader = NHANESCSVLoader(data_path="./data/nhanes/raw_csv")
    feature_extractor = FeatureExtractor()
    disease_labeler = ComprehensiveDiseaseLabeler()
    
    # Get patient cohort
    print(f"\nExtracting {num_patients:,} patients...")
    cohort = loader.get_patient_cohort(max_patients=num_patients, min_age=18, max_age=90)
    print(f"✓ Extracted {len(cohort):,} patients")
    
    # Process patients
    processed_patients = []
    
    print("\nProcessing patients...")
    errors = []
    for i, patient in enumerate(tqdm(cohort)):
        try:
            # Extract features
            graph_features = feature_extractor.extract_graph_features(patient)
            ml_features = feature_extractor.extract_all_features(patient)
            
            # Extract disease labels
            disease_labels = disease_labeler.extract_all_disease_labels(patient)
            
            # Store processed patient
            processed_patients.append({
                'patient_id': patient.get('patient_id', f'P{i:06d}'),
                'age': patient.get('age', 50),
                'sex': patient.get('sex', 'unknown'),
                'bmi': patient.get('bmi', 25),
                'systolic_bp': patient.get('systolic_bp', 120),
                'diastolic_bp': patient.get('diastolic_bp', 80),
                'fasting_glucose': patient.get('fasting_glucose', 100),
                'hba1c': patient.get('hba1c', 5.5),
                'total_cholesterol': patient.get('total_cholesterol', 200),
                'smoking': patient.get('smoking_status') == 'current',
                'exercise_hours_per_week': patient.get('exercise_hours_per_week', 0),
                'graph_features': graph_features,
                'ml_features': ml_features,
                'disease_labels': disease_labels,
                'raw_data': patient
            })
            
        except Exception as e:
            if len(errors) < 5:  # Only store first 5 errors
                errors.append(str(e))
            continue
    
    if errors:
        print(f"\nSample errors (first 5):")
        for err in errors:
            print(f"  - {err}")
    
    print(f"\n✓ Successfully processed {len(processed_patients):,} patients")
    
    # Calculate statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    ages = [p['age'] for p in processed_patients]
    print(f"\nAge: {np.mean(ages):.1f} ± {np.std(ages):.1f} years")
    print(f"Range: {min(ages):.0f}-{max(ages):.0f} years")
    
    sex_counts = {}
    for p in processed_patients:
        sex = p['sex']
        sex_counts[sex] = sex_counts.get(sex, 0) + 1
    print(f"\nSex distribution:")
    for sex, count in sex_counts.items():
        print(f"  {sex}: {count:,} ({count/len(processed_patients)*100:.1f}%)")
    
    # Disease prevalence
    print("\nDisease Prevalence:")
    disease_counts = {}
    for p in processed_patients:
        for disease, has_disease in p['disease_labels'].items():
            if has_disease:
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
        prevalence = count / len(processed_patients) * 100
        print(f"  {disease}: {count:,} ({prevalence:.1f}%)")
    
    # Save data
    output_path = './data/nhanes_multi_disease_10k.pkl'
    print(f"\nSaving to {output_path}...")
    
    data = {
        'patients': processed_patients,
        'metadata': {
            'num_patients': len(processed_patients),
            'num_diseases': 24,
            'num_ml_features': len(processed_patients[0]['ml_features']) if processed_patients else 0,
            'disease_names': list(processed_patients[0]['disease_labels'].keys()) if processed_patients else [],
            'feature_names': feature_extractor.feature_names
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved {len(processed_patients):,} patients to {output_path}")
    print("\nReady for training!")
    
    return data

if __name__ == '__main__':
    process_nhanes_data(num_patients=10000)
