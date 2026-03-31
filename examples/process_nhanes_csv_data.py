#!/usr/bin/env python3
"""
Process NHANES CSV Data for Training
Extracts patient cohort, harmonizes data, and prepares for GNN training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from data_integration.nhanes_csv_loader import NHANESCSVLoader
from data_integration.data_harmonizer import DataHarmonizer
from data_integration.feature_extractor import FeatureExtractor


def main():
    print("=" * 80)
    print("NHANES CSV Data Processing for Digital Twin Training")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing data loaders...")
    nhanes_loader = NHANESCSVLoader(data_path="./data/nhanes/raw_csv")
    harmonizer = DataHarmonizer()
    feature_extractor = FeatureExtractor()
    
    # Load a cohort of patients
    print("\nExtracting patient cohort...")
    print("This may take several minutes for large cohorts...")
    
    # Start with 10,000 patients for initial training
    # You can increase this later
    max_patients = 10000
    
    cohort = nhanes_loader.get_patient_cohort(
        max_patients=max_patients,
        min_age=18,
        max_age=90
    )
    
    if not cohort:
        print("❌ No patients extracted. Check NHANES data files.")
        return
    
    print(f"\n✅ Extracted {len(cohort)} patients")
    
    # Harmonize and extract features
    print("\nHarmonizing data and extracting features...")
    
    processed_data = []
    harmonization_errors = []
    feature_extraction_errors = []
    
    for i, patient_raw in enumerate(cohort):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(cohort)} patients...")
        
        try:
            # Harmonize
            harmonized = harmonizer.harmonize(patient_raw)
            
            # Validate
            errors = harmonizer.validate(harmonized)
            if errors:
                harmonization_errors.append({
                    'patient_id': patient_raw.get('patient_id'),
                    'errors': errors
                })
                continue
            
            # Extract ML features
            try:
                ml_features = feature_extractor.extract_ml_features(harmonized)
                graph_features = feature_extractor.extract_graph_features(harmonized)
                
                processed_data.append({
                    'patient_id': harmonized['patient_id'],
                    'raw': patient_raw,
                    'harmonized': harmonized,
                    'ml_features': ml_features,
                    'graph_features': graph_features
                })
            except Exception as e:
                feature_extraction_errors.append({
                    'patient_id': patient_raw.get('patient_id'),
                    'error': str(e)
                })
        
        except Exception as e:
            harmonization_errors.append({
                'patient_id': patient_raw.get('patient_id'),
                'error': str(e)
            })
    
    print(f"\n✅ Successfully processed {len(processed_data)} patients")
    
    if harmonization_errors:
        print(f"⚠️  Harmonization errors: {len(harmonization_errors)}")
    if feature_extraction_errors:
        print(f"⚠️  Feature extraction errors: {len(feature_extraction_errors)}")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    
    ages = [p['harmonized']['age'] for p in processed_data if 'age' in p['harmonized']]
    bmis = [p['harmonized']['bmi'] for p in processed_data if 'bmi' in p['harmonized'] and p['harmonized']['bmi'] is not None]
    
    print(f"\nDemographics:")
    print(f"  Total patients: {len(processed_data)}")
    print(f"  Age: mean={np.mean(ages):.1f}, min={np.min(ages):.0f}, max={np.max(ages):.0f}")
    
    males = sum(1 for p in processed_data if p['harmonized'].get('sex') == 'M')
    females = sum(1 for p in processed_data if p['harmonized'].get('sex') == 'F')
    print(f"  Sex: {males} male ({100*males/len(processed_data):.1f}%), {females} female ({100*females/len(processed_data):.1f}%)")
    
    if bmis:
        print(f"  BMI: mean={np.mean(bmis):.1f}, min={np.min(bmis):.1f}, max={np.max(bmis):.1f}")
    
    # Disease prevalence
    print(f"\nDisease Prevalence:")
    
    diabetes_count = sum(1 for p in processed_data if p['harmonized'].get('diabetes'))
    hypertension_count = sum(1 for p in processed_data if p['harmonized'].get('hypertension'))
    ckd_count = sum(1 for p in processed_data if p['harmonized'].get('chronic_kidney_disease'))
    cvd_count = sum(1 for p in processed_data if p['harmonized'].get('cardiovascular_disease'))
    
    print(f"  Diabetes: {diabetes_count} ({100*diabetes_count/len(processed_data):.1f}%)")
    print(f"  Hypertension: {hypertension_count} ({100*hypertension_count/len(processed_data):.1f}%)")
    print(f"  CKD: {ckd_count} ({100*ckd_count/len(processed_data):.1f}%)")
    print(f"  CVD: {cvd_count} ({100*cvd_count/len(processed_data):.1f}%)")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    
    if processed_data:
        sample_ml = processed_data[0]['ml_features']
        sample_graph = processed_data[0]['graph_features']
        
        print(f"  ML features per patient: {len(sample_ml)}")
        print(f"  Graph node types: {len(sample_graph)}")
        
        for node_type, features in sample_graph.items():
            print(f"    {node_type}: {len(features)} features")
    
    # Save processed data
    output_file = Path("./data/processed_nhanes_training_data.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving processed data to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(processed_data)} patients ({file_size_mb:.1f} MB)")
    
    # Save error logs if any
    if harmonization_errors or feature_extraction_errors:
        error_log_file = Path("./data/nhanes_processing_errors.pkl")
        with open(error_log_file, 'wb') as f:
            pickle.dump({
                'harmonization_errors': harmonization_errors,
                'feature_extraction_errors': feature_extraction_errors
            }, f)
        print(f"⚠️  Error log saved to {error_log_file}")
    
    # Save summary report
    report_file = Path("./data/nhanes_processing_report.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NHANES Data Processing Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total patients processed: {len(processed_data)}\n")
        f.write(f"Harmonization errors: {len(harmonization_errors)}\n")
        f.write(f"Feature extraction errors: {len(feature_extraction_errors)}\n\n")
        
        f.write("Demographics:\n")
        f.write(f"  Age: mean={np.mean(ages):.1f}, range=[{np.min(ages):.0f}, {np.max(ages):.0f}]\n")
        f.write(f"  Sex: {males} male, {females} female\n")
        if bmis:
            f.write(f"  BMI: mean={np.mean(bmis):.1f}, range=[{np.min(bmis):.1f}, {np.max(bmis):.1f}]\n")
        
        f.write("\nDisease Prevalence:\n")
        f.write(f"  Diabetes: {diabetes_count} ({100*diabetes_count/len(processed_data):.1f}%)\n")
        f.write(f"  Hypertension: {hypertension_count} ({100*hypertension_count/len(processed_data):.1f}%)\n")
        f.write(f"  CKD: {ckd_count} ({100*ckd_count/len(processed_data):.1f}%)\n")
        f.write(f"  CVD: {cvd_count} ({100*cvd_count/len(processed_data):.1f}%)\n")
    
    print(f"📄 Report saved to {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ NHANES Data Processing Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Review the processing report: {report_file}")
    print(f"  2. Train the hybrid GNN model using: python3 train_hybrid_model.py")
    print(f"  3. The processed data is ready at: {output_file}")
    print()


if __name__ == "__main__":
    main()
