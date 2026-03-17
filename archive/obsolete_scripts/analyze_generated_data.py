#!/usr/bin/env python3
"""
Analyze the generated synthetic patient data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_patient_data():
    logger.info("Loading patient data...")
    
    # Load the main patient dataset
    data_file = "data/synthetic/complete_patient_data.csv"
    df = pd.read_csv(data_file)
    
    logger.info(f"\n{'='*80}")
    logger.info("PATIENT POPULATION ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"\nTotal Patients: {len(df):,}")
    
    # Demographics
    logger.info(f"\n--- Demographics ---")
    logger.info(f"Age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
    logger.info(f"Gender: {df['gender'].value_counts().to_dict()}")
    logger.info(f"Ethnicity distribution:")
    for eth, count in df['ethnicity'].value_counts().items():
        logger.info(f"  {eth}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Physical measurements
    logger.info(f"\n--- Physical Measurements ---")
    logger.info(f"BMI: {df['bmi'].mean():.1f} ± {df['bmi'].std():.1f}")
    logger.info(f"  Underweight (<18.5): {(df['bmi'] < 18.5).sum():,} ({(df['bmi'] < 18.5).mean()*100:.1f}%)")
    logger.info(f"  Normal (18.5-25): {((df['bmi'] >= 18.5) & (df['bmi'] < 25)).sum():,} ({((df['bmi'] >= 18.5) & (df['bmi'] < 25)).mean()*100:.1f}%)")
    logger.info(f"  Overweight (25-30): {((df['bmi'] >= 25) & (df['bmi'] < 30)).sum():,} ({((df['bmi'] >= 25) & (df['bmi'] < 30)).mean()*100:.1f}%)")
    logger.info(f"  Obese (≥30): {(df['bmi'] >= 30).sum():,} ({(df['bmi'] >= 30).mean()*100:.1f}%)")
    
    # Vital signs
    logger.info(f"\n--- Vital Signs ---")
    logger.info(f"Systolic BP: {df['systolic_bp'].mean():.1f} ± {df['systolic_bp'].std():.1f} mmHg")
    logger.info(f"Diastolic BP: {df['diastolic_bp'].mean():.1f} ± {df['diastolic_bp'].std():.1f} mmHg")
    logger.info(f"Heart Rate: {df['heart_rate'].mean():.1f} ± {df['heart_rate'].std():.1f} bpm")
    
    # Lab results
    logger.info(f"\n--- Lab Results ---")
    logger.info(f"Glucose: {df['glucose_mmol_l'].mean():.1f} ± {df['glucose_mmol_l'].std():.1f} mmol/L")
    logger.info(f"HbA1c: {df['hba1c_percent'].mean():.1f} ± {df['hba1c_percent'].std():.1f}%")
    logger.info(f"Total Cholesterol: {df['total_cholesterol_mmol_l'].mean():.1f} ± {df['total_cholesterol_mmol_l'].std():.1f} mmol/L")
    logger.info(f"LDL: {df['ldl_cholesterol_mmol_l'].mean():.1f} ± {df['ldl_cholesterol_mmol_l'].std():.1f} mmol/L")
    logger.info(f"HDL: {df['hdl_cholesterol_mmol_l'].mean():.1f} ± {df['hdl_cholesterol_mmol_l'].std():.1f} mmol/L")
    
    # Disease prevalence
    logger.info(f"\n--- Disease Prevalence ---")
    logger.info(f"Hypertension: {df['hypertension'].sum():,} ({df['hypertension'].mean()*100:.1f}%)")
    logger.info(f"Diabetes: {df['diabetes'].sum():,} ({df['diabetes'].mean()*100:.1f}%)")
    logger.info(f"Heart Disease: {df['heart_disease'].sum():,} ({df['heart_disease'].mean()*100:.1f}%)")
    logger.info(f"Stroke: {df['stroke'].sum():,} ({df['stroke'].mean()*100:.1f}%)")
    logger.info(f"Cancer: {df['cancer'].sum():,} ({df['cancer'].mean()*100:.1f}%)")
    logger.info(f"COPD: {df['copd'].sum():,} ({df['copd'].mean()*100:.1f}%)")
    logger.info(f"Depression: {df['depression'].sum():,} ({df['depression'].mean()*100:.1f}%)")
    logger.info(f"Anxiety: {df['anxiety'].sum():,} ({df['anxiety'].mean()*100:.1f}%)")
    
    # Lifestyle factors
    logger.info(f"\n--- Lifestyle Factors ---")
    logger.info(f"Smoking status:")
    for status, count in df['smoking_status'].value_counts().items():
        logger.info(f"  {status}: {count:,} ({count/len(df)*100:.1f}%)")
    logger.info(f"Exercise: {df['exercise_hours_per_week'].mean():.1f} ± {df['exercise_hours_per_week'].std():.1f} hrs/week")
    logger.info(f"Sleep: {df['sleep_hours_per_night'].mean():.1f} ± {df['sleep_hours_per_night'].std():.1f} hrs/night")
    logger.info(f"Alcohol: {df['alcohol_units_per_week'].mean():.1f} ± {df['alcohol_units_per_week'].std():.1f} units/week")
    logger.info(f"Diet Quality: {df['diet_quality_score'].mean():.1f} ± {df['diet_quality_score'].std():.1f} (1-10)")
    logger.info(f"Stress Level: {df['stress_level'].mean():.1f} ± {df['stress_level'].std():.1f} (1-10)")
    
    # Medications
    logger.info(f"\n--- Medications ---")
    logger.info(f"Average medications per patient: {df['medication_count'].mean():.1f}")
    logger.info(f"Patients on medications: {(df['medication_count'] > 0).sum():,} ({(df['medication_count'] > 0).mean()*100:.1f}%)")
    
    # High-risk patients
    logger.info(f"\n--- High-Risk Patients ---")
    high_risk = df[
        (df['bmi'] > 30) & 
        (df['systolic_bp'] > 140) & 
        (df['hba1c_percent'] > 6.0)
    ]
    logger.info(f"Patients with BMI>30, BP>140, and HbA1c>6.0: {len(high_risk):,} ({len(high_risk)/len(df)*100:.1f}%)")
    
    return df


def analyze_trajectories():
    logger.info(f"\n{'='*80}")
    logger.info("DISEASE TRAJECTORY ANALYSIS")
    logger.info(f"{'='*80}")
    
    # Find trajectory files
    traj_files = list(Path("data/synthetic").glob("trajectories_batch_*/disease_trajectories.csv"))
    logger.info(f"\nFound {len(traj_files)} trajectory batch files")
    
    if traj_files:
        # Load first batch as sample
        traj_df = pd.read_csv(traj_files[0])
        
        logger.info(f"\nSample trajectory data (Batch 0):")
        logger.info(f"Total trajectory records: {len(traj_df):,}")
        logger.info(f"Unique patients: {traj_df['patient_id'].nunique():,}")
        
        # Final year outcomes
        final_year = traj_df[traj_df['year'] == 10]
        logger.info(f"\n10-Year Outcomes:")
        logger.info(f"  Diabetes: {final_year['diabetes'].mean()*100:.1f}%")
        logger.info(f"  CVD: {final_year['cardiovascular_disease'].mean()*100:.1f}%")
        logger.info(f"  Cancer: {final_year['cancer'].mean()*100:.1f}%")
        logger.info(f"  Survival: {final_year['alive'].mean()*100:.1f}%")


if __name__ == "__main__":
    df = analyze_patient_data()
    analyze_trajectories()
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Analysis complete!")
    logger.info(f"{'='*80}")
