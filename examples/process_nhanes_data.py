#!/usr/bin/env python3
"""
Process NHANES Data for Training
Extract and harmonize features from all NHANES patients
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_integration.nhanes_loader import NHANESLoader
from data_integration.data_harmonizer import DataHarmonizer
from data_integration.feature_extractor import FeatureExtractor


def process_nhanes_data(
    output_file: str = './data/nhanes_processed.pkl',
    max_patients: int = 5000
):
    """
    Process NHANES data for training
    
    Args:
        output_file: Where to save processed data
        max_patients: Maximum number of patients to process
    """
    
    print("=" * 80)
    print("PROCESSING NHANES DATA FOR TRAINING")
    print("=" * 80)
    
    # Initialize
    print("\n1. Initializing loaders...")
    loader = NHANESLoader('./data/nhanes', cycle='2017-2018')
    harmonizer = DataHarmonizer()
    extractor = FeatureExtractor()
    
    # Get cohort
    print(f"\n2. Getting cohort (max {max_patients} patients)...")
    cohort = loader.get_cohort(
        min_age=18,
        max_age=90,
        has_labs=True,
        limit=max_patients
    )
    
    print(f"   Found {len(cohort)} patients")
    
    # Process all patients
    print(f"\n3. Processing patients...")
    
    processed_data = []
    skipped = 0
    
    for seqn in tqdm(cohort, desc="Processing"):
        try:
            # Extract raw features
            raw_features = loader.extract_patient_features(seqn)
            
            # Harmonize
            harmonized = harmonizer.harmonize(raw_features, source='nhanes')
            
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
                }
            })
        
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\n4. Processing complete:")
    print(f"   ✅ Successfully processed: {len(processed_data)} patients")
    print(f"   ⚠️  Skipped (incomplete data): {skipped} patients")
    
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
    
    # Save processed data
    print(f"\n6. Saving processed data...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"   ✅ Saved to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Also save as CSV for inspection
    csv_path = output_path.with_suffix('.csv')
    df_data = []
    for p in processed_data:
        row = {
            'patient_id': p['patient_id'],
            **p['demographics'],
            **p['labels'],
            **p['raw_values']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    print(f"   ✅ Also saved CSV to {csv_path}")
    
    print(f"\n" + "=" * 80)
    print(f"✅ DATA PROCESSING COMPLETE")
    print(f"=" * 80)
    print(f"\nNext step:")
    print(f"python3 train_hybrid_model.py --data {output_file}")
    
    return processed_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process NHANES data')
    parser.add_argument('--output', default='./data/nhanes_processed.pkl',
                       help='Output file path')
    parser.add_argument('--max-patients', type=int, default=5000,
                       help='Maximum number of patients to process')
    
    args = parser.parse_args()
    
    process_nhanes_data(args.output, args.max_patients)
