#!/usr/bin/env python3
"""
Process Synthetic Data for Training
Load synthetic patient data and prepare for hybrid GNN training
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_integration.data_harmonizer import DataHarmonizer
from data_integration.feature_extractor import FeatureExtractor


def process_synthetic_data(
    input_file: str = './data/training_data.csv',
    output_file: str = './data/processed_training_data.pkl'
):
    """
    Process synthetic data for training
    
    Args:
        input_file: CSV file with synthetic patients
        output_file: Where to save processed data
    """
    
    print("=" * 80)
    print("PROCESSING SYNTHETIC DATA FOR TRAINING")
    print("=" * 80)
    
    # Load synthetic data
    print(f"\n1. Loading synthetic data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} patients")
    
    # Initialize processors
    print(f"\n2. Initializing data processors...")
    harmonizer = DataHarmonizer()
    extractor = FeatureExtractor()
    
    # Process all patients
    print(f"\n3. Processing patients...")
    
    processed_data = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            # Convert row to dictionary
            patient_dict = row.to_dict()
            
            # Harmonize (synthetic data is already in good format)
            harmonized = harmonizer.harmonize(patient_dict, source='manual')
            
            # Validate
            is_valid, errors = harmonizer.validate(harmonized)
            if not is_valid:
                skipped += 1
                continue
            
            # Extract ML features
            ml_features = extractor.extract_all_features(harmonized)
            
            # Extract graph features (for GNN)
            graph_features = extractor.extract_graph_features(harmonized)
            
            # Store
            processed_data.append({
                'patient_id': harmonized['patient_id'],
                'demographics': {
                    'age': harmonized.get('age'),
                    'sex': harmonized.get('sex'),
                },
                'features': ml_features,
                'feature_names': extractor.get_feature_names(),
                'graph_features': graph_features,
                'labels': {
                    'has_diabetes': harmonized.get('has_diabetes', False),
                    'has_hypertension': harmonized.get('has_hypertension', False),
                    'has_ckd': harmonized.get('has_ckd', False),
                },
                'raw_values': {
                    'glucose': harmonized.get('fasting_glucose'),
                    'hba1c': harmonized.get('hba1c'),
                    'systolic_bp': harmonized.get('systolic_bp'),
                    'diastolic_bp': harmonized.get('diastolic_bp'),
                    'bmi': harmonized.get('bmi'),
                    'ldl': harmonized.get('ldl'),
                    'hdl': harmonized.get('hdl'),
                    'triglycerides': harmonized.get('triglycerides'),
                    'creatinine': harmonized.get('creatinine'),
                    'egfr': harmonized.get('egfr'),
                    'alt': harmonized.get('alt'),
                    'ast': harmonized.get('ast'),
                    'crp': harmonized.get('crp'),
                }
            })
        
        except Exception as e:
            print(f"\nError processing patient {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\n4. Processing complete:")
    print(f"   ✅ Successfully processed: {len(processed_data)} patients")
    print(f"   ⚠️  Skipped: {skipped} patients")
    
    # Calculate statistics
    print(f"\n5. Dataset statistics:")
    
    diabetes_count = sum(1 for p in processed_data if p['labels']['has_diabetes'])
    hypertension_count = sum(1 for p in processed_data if p['labels']['has_hypertension'])
    ckd_count = sum(1 for p in processed_data if p['labels']['has_ckd'])
    
    print(f"   Diabetes prevalence: {diabetes_count}/{len(processed_data)} ({100*diabetes_count/len(processed_data):.1f}%)")
    print(f"   Hypertension prevalence: {hypertension_count}/{len(processed_data)} ({100*hypertension_count/len(processed_data):.1f}%)")
    print(f"   CKD prevalence: {ckd_count}/{len(processed_data)} ({100*ckd_count/len(processed_data):.1f}%)")
    
    # Age distribution
    ages = [p['demographics']['age'] for p in processed_data if p['demographics']['age']]
    if ages:
        print(f"   Age range: {min(ages)} - {max(ages)} years")
        print(f"   Mean age: {sum(ages)/len(ages):.1f} years")
    
    # Sex distribution
    males = sum(1 for p in processed_data if p['demographics']['sex'] == 'M')
    females = sum(1 for p in processed_data if p['demographics']['sex'] == 'F')
    print(f"   Sex: {males} male ({100*males/len(processed_data):.1f}%), {females} female ({100*females/len(processed_data):.1f}%)")
    
    # Feature statistics
    if processed_data:
        print(f"   Features per patient: {len(processed_data[0]['features'])}")
        print(f"   Graph node types: {len(processed_data[0]['graph_features'])}")
    
    # Save processed data
    print(f"\n6. Saving processed data...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"   ✅ Saved to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n" + "=" * 80)
    print(f"✅ DATA PROCESSING COMPLETE")
    print(f"=" * 80)
    print(f"\nReady for training!")
    print(f"Next step: python3 train_hybrid_model.py")
    
    return processed_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process synthetic data')
    parser.add_argument('--input', default='./data/training_data.csv',
                       help='Input CSV file')
    parser.add_argument('--output', default='./data/processed_training_data.pkl',
                       help='Output pickle file')
    
    args = parser.parse_args()
    
    process_synthetic_data(args.input, args.output)
