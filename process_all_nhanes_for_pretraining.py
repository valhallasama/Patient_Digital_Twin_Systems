#!/usr/bin/env python3
"""
Process ALL NHANES Patients for Stage 1 Pretraining (135K patients)

Unlike the standard processing, this includes patients with incomplete data
since masked pretraining can handle missing values.
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
import pandas as pd

def process_all_nhanes_for_pretraining(min_age: int = None, max_age: int = None):
    """
    Process ALL NHANES patients for pretraining, including incomplete data
    """
    
    print("="*80)
    print("PROCESSING ALL NHANES PATIENTS FOR STAGE 1 PRETRAINING")
    print("="*80)
    
    # Initialize
    loader = NHANESCSVLoader(data_path="./data/nhanes/raw_csv")
    feature_extractor = FeatureExtractor()
    disease_labeler = ComprehensiveDiseaseLabeler()
    
    # Load demographics directly to get ALL patient IDs
    print(f"\n🚀 Loading ALL patients (no age filter)...")
    demo = pd.read_csv('./data/nhanes/raw_csv/demographics_clean.csv')
    
    print(f"Total patients in demographics: {len(demo):,}")
    
    # NO age filtering - use ALL patients
    print(f"Using all {len(demo):,} patients (all ages)")
    
    # Get ALL patient IDs (including duplicates from different survey cycles)
    # Each survey cycle is treated as a separate observation for pretraining
    if 'SEQN' not in demo.columns:
        print("ERROR: SEQN column not found")
        return
    
    patient_ids = demo['SEQN'].values  # Keep ALL entries, including duplicates
    total_patients = len(patient_ids)
    
    print(f"✓ Found {total_patients:,} patient observations (may include same patient across survey cycles)")
    
    # Process ALL patients (even with incomplete data)
    processed_patients = []
    failed_count = 0
    incomplete_count = 0
    
    print("\nProcessing patients (allowing incomplete data)...")
    for i, seqn in enumerate(tqdm(patient_ids, desc="Processing")):
        try:
            # Extract patient features (may be incomplete)
            patient = loader.extract_patient_features(seqn)
            
            if not patient:
                failed_count += 1
                continue
            
            # Extract graph features (with imputation for missing values)
            graph_features = feature_extractor.extract_graph_features(patient)
            
            # Verify all organs present (imputation should fill missing)
            expected_organs = ['metabolic', 'cardiovascular', 'liver', 'kidney', 'immune', 'neural', 'lifestyle']
            if not all(organ in graph_features for organ in expected_organs):
                failed_count += 1
                continue
            
            # Check for NaN in graph features
            has_nan = False
            for organ, features in graph_features.items():
                if np.isnan(features).any():
                    has_nan = True
                    break
            
            if has_nan:
                failed_count += 1
                continue
            
            # Extract disease labels (may be incomplete - that's OK for pretraining)
            disease_labels = disease_labeler.extract_all_disease_labels(patient)
            
            # Check if patient has complete disease labels
            has_complete_labels = all(
                label is not None 
                for label in disease_labels.values()
            )
            
            if not has_complete_labels:
                incomplete_count += 1
            
            # Store processed patient
            processed_patients.append({
                'patient_id': patient.get('patient_id', f'P{i:06d}'),
                'age': patient.get('age', 50),
                'sex': patient.get('sex', 'unknown'),
                'graph_features': graph_features,
                'disease_labels': disease_labels,
                'has_complete_labels': has_complete_labels  # Flag for Stage 2 filtering
            })
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"\nError processing patient {seqn}: {e}")
            continue
    
    print(f"\n✓ Successfully processed: {len(processed_patients):,} patients")
    print(f"  - Complete disease labels: {len(processed_patients) - incomplete_count:,}")
    print(f"  - Incomplete disease labels: {incomplete_count:,}")
    print(f"  - Failed: {failed_count:,}")
    
    # Save to pickle
    output_file = f'./data/nhanes_all_{len(processed_patients)}.pkl'
    print(f"\n💾 Saving to {output_file}...")
    
    data = {
        'patients': processed_patients,
        'metadata': {
            'total_patients': len(processed_patients),
            'complete_labels': len(processed_patients) - incomplete_count,
            'incomplete_labels': incomplete_count,
            'all_ages': True,
            'processing_date': pd.Timestamp.now().isoformat()
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Saved {len(processed_patients):,} patients")
    print(f"\nDataset ready for two-stage training:")
    print(f"  Stage 1 (Pretraining): Use all {len(processed_patients):,} patients")
    print(f"  Stage 2 (Fine-tuning): Use {len(processed_patients) - incomplete_count:,} complete patients")
    print("="*80)

if __name__ == '__main__':
    # Process ALL patients without age filtering
    process_all_nhanes_for_pretraining()
