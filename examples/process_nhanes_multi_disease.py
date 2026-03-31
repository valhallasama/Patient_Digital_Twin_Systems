"""
Process NHANES Data for Multi-Disease Prediction

Extracts comprehensive disease labels for all 24 diseases from real NHANES data.
Creates training dataset for comprehensive digital twin system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_integration.nhanes_csv_loader import NHANESCSVLoader
from data_integration.data_harmonizer import DataHarmonizer
from data_integration.feature_extractor import FeatureExtractor
from data_integration.comprehensive_disease_labels import ComprehensiveDiseaseLabeler
import pandas as pd
import numpy as np
import pickle
from datetime import datetime


def process_nhanes_cohort(num_patients: int = 10000, 
                          min_age: int = 18, 
                          max_age: int = 90,
                          save_path: str = './data/nhanes_multi_disease_10k.pkl'):
    """
    Process NHANES cohort with comprehensive disease labels
    
    Args:
        num_patients: Number of patients to extract
        min_age: Minimum age
        max_age: Maximum age
        save_path: Where to save processed data
    """
    
    print("=" * 80)
    print("NHANES MULTI-DISEASE DATA PROCESSING")
    print("=" * 80)
    print(f"\nTarget: {num_patients:,} patients")
    print(f"Age range: {min_age}-{max_age} years")
    print(f"Diseases: 24 comprehensive disease labels")
    print()
    
    # Initialize components
    print("Initializing data loaders...")
    loader = NHANESCSVLoader(data_path="./data/nhanes/raw_csv")
    harmonizer = DataHarmonizer()
    feature_extractor = FeatureExtractor()
    disease_labeler = ComprehensiveDiseaseLabeler()
    
    # Load NHANES cohort
    print(f"\nExtracting {num_patients:,} patients from NHANES...")
    cohort = loader.get_patient_cohort(
        max_patients=num_patients,
        min_age=min_age,
        max_age=max_age
    )
    
    print(f"✓ Extracted {len(cohort):,} patients with complete data")
    
    # Process each patient
    print("\nProcessing patients...")
    processed_data = []
    errors = []
    
    for i, patient in enumerate(cohort):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1:,}/{len(cohort):,} patients...")
        
        try:
            # Harmonize data
            harmonized = harmonizer.harmonize(patient, source='NHANES')
            
            # Extract ML features
            ml_features = feature_extractor.extract_all_features(harmonized)
            
            # Extract disease labels (all 24 diseases)
            disease_labels = disease_labeler.extract_all_disease_labels(harmonized)
            
            # Extract graph features
            graph_features = feature_extractor.extract_graph_features(harmonized)
            
            processed_data.append({
                'patient_id': harmonized.get('patient_id'),
                'source': 'NHANES',
                'demographics': {
                    'age': harmonized.get('age'),
                    'sex': harmonized.get('sex'),
                    'race': harmonized.get('race_ethnicity')
                },
                'ml_features': ml_features,
                'feature_names': feature_extractor.feature_names,
                'graph_features': graph_features,
                'disease_labels': disease_labels,
                'raw_data': harmonized
            })
            
        except Exception as e:
            errors.append({
                'patient_id': patient.get('patient_id'),
                'error': str(e)
            })
    
    print(f"\n✓ Successfully processed {len(processed_data):,} patients")
    if errors:
        print(f"✗ Failed to process {len(errors)} patients")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("COHORT STATISTICS")
    print("=" * 80)
    
    ages = [p['demographics']['age'] for p in processed_data if p['demographics']['age']]
    sexes = [p['demographics']['sex'] for p in processed_data if p['demographics']['sex']]
    
    print(f"\nDemographics:")
    print(f"  Total patients: {len(processed_data):,}")
    print(f"  Age: {np.mean(ages):.1f} ± {np.std(ages):.1f} years")
    print(f"  Age range: {min(ages):.0f}-{max(ages):.0f} years")
    print(f"  Male: {sexes.count('male'):,} ({100*sexes.count('male')/len(sexes):.1f}%)")
    print(f"  Female: {sexes.count('female'):,} ({100*sexes.count('female')/len(sexes):.1f}%)")
    
    # Disease prevalence
    print("\n" + "=" * 80)
    print("DISEASE PREVALENCE")
    print("=" * 80)
    
    raw_cohort = [p['raw_data'] for p in processed_data]
    prevalence = disease_labeler.calculate_disease_prevalence(raw_cohort)
    
    categories = disease_labeler.get_disease_categories()
    
    for category, diseases in categories.items():
        print(f"\n{category}:")
        for disease in diseases:
            prev = prevalence[disease]
            count = int(prev * len(processed_data))
            print(f"  {disease:30} {count:5,} ({100*prev:5.1f}%)")
    
    # Overall disease burden
    disease_counts = [disease_labeler.get_disease_count(p['raw_data']) for p in processed_data]
    print(f"\nOverall Disease Burden:")
    print(f"  Mean diseases per patient: {np.mean(disease_counts):.2f}")
    print(f"  Median diseases per patient: {np.median(disease_counts):.0f}")
    print(f"  Max diseases in one patient: {max(disease_counts)}")
    
    # Feature statistics
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)
    
    print(f"\nML Features: {len(processed_data[0]['ml_features'])} features")
    print(f"Graph Nodes: {len(processed_data[0]['graph_features']['nodes'])} nodes")
    print(f"Graph Edges: {len(processed_data[0]['graph_features']['edges'])} edge types")
    print(f"Disease Labels: {len(processed_data[0]['disease_labels'])} diseases")
    
    # Save processed data
    print("\n" + "=" * 80)
    print("SAVING DATA")
    print("=" * 80)
    
    output_data = {
        'patients': processed_data,
        'metadata': {
            'source': 'NHANES 1988-2018',
            'num_patients': len(processed_data),
            'num_features': len(processed_data[0]['ml_features']),
            'feature_names': processed_data[0]['feature_names'],
            'num_diseases': len(processed_data[0]['disease_labels']),
            'disease_names': list(processed_data[0]['disease_labels'].keys()),
            'disease_prevalence': prevalence,
            'age_range': (min(ages), max(ages)),
            'processing_date': datetime.now().isoformat(),
            'errors': errors
        }
    }
    
    print(f"\nSaving to: {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    file_size_mb = Path(save_path).stat().st_size / (1024 * 1024)
    print(f"✓ Saved {file_size_mb:.1f} MB")
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    print(f"\n✓ Processed {len(processed_data):,} patients")
    print(f"✓ Extracted {len(processed_data[0]['ml_features'])} ML features")
    print(f"✓ Labeled {len(processed_data[0]['disease_labels'])} diseases")
    print(f"✓ Created graph structure with {len(processed_data[0]['graph_features']['nodes'])} nodes")
    print(f"✓ Data saved to: {save_path}")
    
    print("\nReady for multi-disease prediction training!")
    
    return output_data


def main():
    """Main processing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process NHANES data for multi-disease prediction')
    parser.add_argument('--num_patients', type=int, default=10000,
                       help='Number of patients to process (default: 10000)')
    parser.add_argument('--min_age', type=int, default=18,
                       help='Minimum age (default: 18)')
    parser.add_argument('--max_age', type=int, default=90,
                       help='Maximum age (default: 90)')
    parser.add_argument('--output', type=str, default='./data/nhanes_multi_disease_10k.pkl',
                       help='Output file path')
    
    args = parser.parse_args()
    
    try:
        data = process_nhanes_cohort(
            num_patients=args.num_patients,
            min_age=args.min_age,
            max_age=args.max_age,
            save_path=args.output
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Train multi-disease prediction model:")
        print("   python3 train_multi_disease_gnn.py")
        print("\n2. Run digital twin simulation:")
        print("   python3 simulate_patient_digital_twin.py")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
