#!/usr/bin/env python3
"""
Process FULL NHANES Dataset (135K patients) for GNN-Transformer Training

Uses all available NHANES data for optimal Transformer performance.
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
import argparse

def process_full_nhanes(max_patients: int = None, min_age: int = 18, max_age: int = 90):
    """Process full NHANES dataset"""
    
    print("="*80)
    print("PROCESSING FULL NHANES DATASET FOR GNN-TRANSFORMER")
    print("="*80)
    
    # Initialize
    loader = NHANESCSVLoader(data_path="./data/nhanes/raw_csv")
    feature_extractor = FeatureExtractor()
    disease_labeler = ComprehensiveDiseaseLabeler()
    
    # Get ALL patients (or specified max)
    if max_patients is None:
        print(f"\n🚀 Extracting ALL available patients (age {min_age}-{max_age})...")
        # Use a large number to get all
        max_patients = 200000
    else:
        print(f"\nExtracting up to {max_patients:,} patients (age {min_age}-{max_age})...")
    
    cohort = loader.get_patient_cohort(
        max_patients=max_patients, 
        min_age=min_age, 
        max_age=max_age
    )
    
    total_patients = len(cohort)
    print(f"✓ Extracted {total_patients:,} patients with complete data")
    
    # Process patients
    processed_patients = []
    failed_count = 0
    
    print("\nProcessing patients...")
    for i, patient in enumerate(tqdm(cohort, desc="Processing")):
        try:
            # Extract graph features
            graph_features = feature_extractor.extract_graph_features(patient)
            
            # Verify all organs present
            expected_organs = ['metabolic', 'cardiovascular', 'liver', 'kidney', 'immune', 'neural', 'lifestyle']
            if not all(organ in graph_features for organ in expected_organs):
                failed_count += 1
                continue
            
            # Extract ML features
            ml_features = feature_extractor.extract_all_features(patient)
            
            # Extract disease labels
            disease_labels = disease_labeler.extract_all_disease_labels(patient)
            
            # Store processed patient
            processed_patients.append({
                'patient_id': patient.get('patient_id', f'P{i:06d}'),
                'age': patient.get('age', 50),
                'sex': patient.get('sex', 'unknown'),
                'bmi': patient.get('bmi', 25.0),
                'systolic_bp': patient.get('systolic_bp', 120.0),
                'diastolic_bp': patient.get('diastolic_bp', 80.0),
                'fasting_glucose': patient.get('fasting_glucose', 100.0),
                'hba1c': patient.get('hba1c', 5.5),
                'total_cholesterol': patient.get('total_cholesterol', 200.0),
                'smoking': patient.get('smoking_status') == 'current',
                'exercise_hours_per_week': patient.get('exercise_hours_per_week', 0.0),
                'graph_features': graph_features,
                'ml_features': ml_features,
                'disease_labels': disease_labels,
                'raw_data': patient
            })
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:  # Print first 3 errors with traceback
                import traceback
                print(f"\nError processing patient {i}: {e}")
                print(f"Patient data keys: {list(patient.keys())[:10]}")
                traceback.print_exc()
            continue
    
    success_count = len(processed_patients)
    print(f"\n✓ Successfully processed {success_count:,} patients")
    print(f"✗ Failed: {failed_count:,} patients ({failed_count/total_patients*100:.1f}%)")
    
    if success_count == 0:
        print("\n❌ ERROR: No patients successfully processed!")
        return None
    
    # Calculate statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    ages = [p['age'] for p in processed_patients]
    print(f"\nTotal Patients: {success_count:,}")
    print(f"Age: {np.mean(ages):.1f} ± {np.std(ages):.1f} years")
    print(f"Range: {min(ages):.0f}-{max(ages):.0f} years")
    
    # Sex distribution
    sex_counts = {}
    for p in processed_patients:
        sex = p['sex']
        sex_counts[sex] = sex_counts.get(sex, 0) + 1
    
    print(f"\nSex Distribution:")
    for sex, count in sorted(sex_counts.items()):
        print(f"  {sex}: {count:,} ({count/success_count*100:.1f}%)")
    
    # Disease prevalence
    print("\nDisease Prevalence (Top 15):")
    disease_counts = {}
    for p in processed_patients:
        for disease, has_disease in p['disease_labels'].items():
            if has_disease:
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    for disease, count in sorted_diseases[:15]:
        prevalence = count / success_count * 100
        print(f"  {disease:30s}: {count:6,} ({prevalence:5.1f}%)")
    
    # Feature statistics
    print(f"\nFeatures:")
    print(f"  ML features: {len(processed_patients[0]['ml_features'])}")
    print(f"  Graph organs: {len(processed_patients[0]['graph_features'])}")
    print(f"  Disease labels: {len(processed_patients[0]['disease_labels'])}")
    
    # Save data
    output_path = f'./data/nhanes_full_{success_count}.pkl'
    print(f"\nSaving to {output_path}...")
    
    data = {
        'patients': processed_patients,
        'metadata': {
            'num_patients': success_count,
            'num_diseases': 24,
            'num_ml_features': len(processed_patients[0]['ml_features']),
            'disease_names': list(processed_patients[0]['disease_labels'].keys()),
            'feature_names': feature_extractor.feature_names,
            'organ_names': list(processed_patients[0]['graph_features'].keys()),
            'age_range': (min(ages), max(ages)),
            'processing_date': str(np.datetime64('today'))
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ Saved {success_count:,} patients to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # Recommendations
    print("\n" + "="*80)
    print("TRAINING RECOMMENDATIONS")
    print("="*80)
    
    if success_count >= 100000:
        print("\n✅ EXCELLENT: 100K+ patients - Optimal for Transformer training")
        print("   Recommended config:")
        print("   - Batch size: 64-128")
        print("   - Pretraining epochs: 15-20")
        print("   - Training epochs: 100+")
        print("   - Expected mean AUC: 0.86-0.90")
    elif success_count >= 50000:
        print("\n✅ GOOD: 50K+ patients - Sufficient for Transformer")
        print("   Recommended config:")
        print("   - Batch size: 32-64")
        print("   - Pretraining epochs: 10-15")
        print("   - Training epochs: 50-100")
        print("   - Expected mean AUC: 0.84-0.88")
    elif success_count >= 20000:
        print("\n⚠️  MODERATE: 20K+ patients - Minimum for Transformer")
        print("   Recommended config:")
        print("   - Batch size: 32")
        print("   - Pretraining epochs: 10")
        print("   - Training epochs: 50")
        print("   - Expected mean AUC: 0.82-0.86")
    else:
        print("\n❌ WARNING: <20K patients - Too small for Transformer")
        print("   Consider using GNN-only or collecting more data")
    
    print(f"\n🚀 Ready for training with {success_count:,} patients!")
    print(f"   Run: python3 train_gnn_transformer.py --data_path {output_path}")
    
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process NHANES data for training')
    parser.add_argument('--max_patients', type=int, default=None,
                      help='Maximum patients to process (default: all available)')
    parser.add_argument('--min_age', type=int, default=18,
                      help='Minimum age (default: 18)')
    parser.add_argument('--max_age', type=int, default=90,
                      help='Maximum age (default: 90)')
    
    args = parser.parse_args()
    
    process_full_nhanes(
        max_patients=args.max_patients,
        min_age=args.min_age,
        max_age=args.max_age
    )
