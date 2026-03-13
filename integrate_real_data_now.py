#!/usr/bin/env python3
"""
Integrate Existing Real Data into Digital Twin System
Use diabetes readmission and heart disease datasets we already have
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_diabetes_data():
    """Load diabetes readmission dataset (100k patients)"""
    print("\n📊 Loading Diabetes Readmission Data...")
    
    data_path = Path("data/real/raw/dataset_diabetes/diabetic_data.csv")
    if not data_path.exists():
        print(f"❌ Diabetes data not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} diabetes patient records")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Features: {', '.join(df.columns[:10])}...")
    
    return df


def load_heart_disease_data():
    """Load heart disease datasets"""
    print("\n❤️  Loading Heart Disease Data...")
    
    datasets = []
    
    # UCI dataset
    uci_path = Path("data/real/raw/heart_disease_uci.csv")
    if uci_path.exists():
        uci_df = pd.read_csv(uci_path)
        print(f"✓ UCI dataset: {len(uci_df):,} patients")
        datasets.append(uci_df)
    
    # Hungarian dataset
    hungarian_path = Path("data/real/raw/heart_disease_hungarian.csv")
    if hungarian_path.exists():
        hungarian_df = pd.read_csv(hungarian_path)
        print(f"✓ Hungarian dataset: {len(hungarian_df):,} patients")
        datasets.append(hungarian_df)
    
    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        print(f"✓ Combined: {len(combined):,} total heart disease patients")
        return combined
    
    return None


def extract_metabolic_parameters(diabetes_df):
    """
    Extract metabolic parameters from real diabetes data
    For MetabolicAgent in organ_agents.py
    """
    print("\n🔬 Extracting Metabolic Parameters from Real Data...")
    
    params = {}
    
    # HbA1c progression (if available)
    if 'A1Cresult' in diabetes_df.columns:
        a1c_map = {'>8': 8.5, '>7': 7.5, 'Norm': 5.5, 'None': np.nan}
        diabetes_df['a1c_numeric'] = diabetes_df['A1Cresult'].map(a1c_map)
        
        mean_a1c = diabetes_df['a1c_numeric'].mean()
        std_a1c = diabetes_df['a1c_numeric'].std()
        
        params['hba1c_mean'] = mean_a1c
        params['hba1c_std'] = std_a1c
        print(f"  • HbA1c: {mean_a1c:.2f} ± {std_a1c:.2f}%")
    
    # Readmission rate (proxy for disease progression)
    if 'readmitted' in diabetes_df.columns:
        readmit_30 = (diabetes_df['readmitted'] == '<30').mean()
        readmit_any = (diabetes_df['readmitted'] != 'NO').mean()
        
        params['readmission_30day'] = readmit_30
        params['readmission_any'] = readmit_any
        print(f"  • 30-day readmission: {readmit_30:.1%}")
        print(f"  • Any readmission: {readmit_any:.1%}")
    
    # Number of medications (proxy for disease severity)
    if 'num_medications' in diabetes_df.columns:
        med_mean = diabetes_df['num_medications'].mean()
        med_std = diabetes_df['num_medications'].std()
        
        params['medications_mean'] = med_mean
        params['medications_std'] = med_std
        print(f"  • Medications: {med_mean:.1f} ± {med_std:.1f}")
    
    # Time in hospital (disease burden)
    if 'time_in_hospital' in diabetes_df.columns:
        time_mean = diabetes_df['time_in_hospital'].mean()
        time_std = diabetes_df['time_in_hospital'].std()
        
        params['hospital_days_mean'] = time_mean
        params['hospital_days_std'] = time_std
        print(f"  • Hospital days: {time_mean:.1f} ± {time_std:.1f}")
    
    # Age distribution
    if 'age' in diabetes_df.columns:
        # Age is in brackets like [50-60)
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        diabetes_df['age_numeric'] = diabetes_df['age'].map(age_map)
        
        age_mean = diabetes_df['age_numeric'].mean()
        params['age_mean'] = age_mean
        print(f"  • Mean age: {age_mean:.1f} years")
    
    return params


def extract_cardiovascular_parameters(heart_df):
    """
    Extract cardiovascular parameters from real heart disease data
    For CardiovascularAgent in organ_agents.py
    """
    print("\n❤️  Extracting Cardiovascular Parameters from Real Data...")
    
    params = {}
    
    # Blood pressure (if available)
    if 'trestbps' in heart_df.columns:
        bp_mean = heart_df['trestbps'].mean()
        bp_std = heart_df['trestbps'].std()
        
        params['resting_bp_mean'] = bp_mean
        params['resting_bp_std'] = bp_std
        print(f"  • Resting BP: {bp_mean:.1f} ± {bp_std:.1f} mmHg")
    
    # Cholesterol
    if 'chol' in heart_df.columns:
        chol_mean = heart_df['chol'].mean()
        chol_std = heart_df['chol'].std()
        
        params['cholesterol_mean'] = chol_mean
        params['cholesterol_std'] = chol_std
        print(f"  • Cholesterol: {chol_mean:.1f} ± {chol_std:.1f} mg/dL")
    
    # Max heart rate
    if 'thalach' in heart_df.columns:
        hr_mean = heart_df['thalach'].mean()
        hr_std = heart_df['thalach'].std()
        
        params['max_hr_mean'] = hr_mean
        params['max_hr_std'] = hr_std
        print(f"  • Max heart rate: {hr_mean:.1f} ± {hr_std:.1f} bpm")
    
    # Disease prevalence
    if 'target' in heart_df.columns:
        disease_rate = heart_df['target'].mean()
        params['disease_prevalence'] = disease_rate
        print(f"  • Heart disease prevalence: {disease_rate:.1%}")
    
    # Age distribution
    if 'age' in heart_df.columns:
        age_mean = heart_df['age'].mean()
        age_std = heart_df['age'].std()
        
        params['age_mean'] = age_mean
        params['age_std'] = age_std
        print(f"  • Age: {age_mean:.1f} ± {age_std:.1f} years")
    
    return params


def train_diabetes_readmission_model(diabetes_df):
    """Train model to predict diabetes readmission"""
    print("\n🤖 Training Diabetes Readmission Model...")
    
    # Prepare features
    feature_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]
    
    # Check which features exist
    available_features = [col for col in feature_cols if col in diabetes_df.columns]
    
    if not available_features:
        print("❌ No numeric features available for training")
        return None
    
    # Create target (readmitted within 30 days)
    if 'readmitted' not in diabetes_df.columns:
        print("❌ No readmission target available")
        return None
    
    df_clean = diabetes_df[available_features + ['readmitted']].copy()
    df_clean = df_clean.dropna()
    
    X = df_clean[available_features]
    y = (df_clean['readmitted'] == '<30').astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"✓ Model trained on {len(X_train):,} patients")
    print(f"  • Accuracy: {accuracy:.3f}")
    print(f"  • AUC: {auc:.3f}")
    
    # Save model
    model_path = Path("models/real_data/diabetes_readmission_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': available_features,
            'accuracy': accuracy,
            'auc': auc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }, f)
    
    print(f"✓ Model saved to {model_path}")
    
    return model


def train_heart_disease_model(heart_df):
    """Train model to predict heart disease"""
    print("\n❤️  Training Heart Disease Model...")
    
    # Prepare features
    feature_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Check which features exist
    available_features = [col for col in feature_cols if col in heart_df.columns]
    
    if not available_features:
        print("❌ No features available for training")
        return None
    
    if 'target' not in heart_df.columns:
        print("❌ No target variable available")
        return None
    
    df_clean = heart_df[available_features + ['target']].copy()
    df_clean = df_clean.dropna()
    
    X = df_clean[available_features]
    y = df_clean['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"✓ Model trained on {len(X_train):,} patients")
    print(f"  • Accuracy: {accuracy:.3f}")
    print(f"  • AUC: {auc:.3f}")
    
    # Save model
    model_path = Path("models/real_data/heart_disease_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': available_features,
            'accuracy': accuracy,
            'auc': auc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }, f)
    
    print(f"✓ Model saved to {model_path}")
    
    return model


def save_parameters_to_json(metabolic_params, cardiovascular_params):
    """Save extracted parameters for use in organ agents"""
    print("\n💾 Saving Parameters...")
    
    all_params = {
        'metabolic': metabolic_params,
        'cardiovascular': cardiovascular_params,
        'source': 'real_patient_data',
        'datasets': {
            'diabetes': 'UCI Diabetes 130-US hospitals (100k patients)',
            'heart_disease': 'UCI Heart Disease (Cleveland + Hungarian)'
        },
        'extraction_date': pd.Timestamp.now().isoformat()
    }
    
    params_path = Path("data/real/extracted_parameters.json")
    with open(params_path, 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"✓ Parameters saved to {params_path}")
    
    return all_params


def main():
    print("="*80)
    print("INTEGRATE EXISTING REAL DATA INTO DIGITAL TWIN")
    print("="*80)
    
    # Load datasets
    diabetes_df = load_diabetes_data()
    heart_df = load_heart_disease_data()
    
    if diabetes_df is None and heart_df is None:
        print("\n❌ No real data found!")
        return
    
    # Extract parameters
    metabolic_params = {}
    cardiovascular_params = {}
    
    if diabetes_df is not None:
        metabolic_params = extract_metabolic_parameters(diabetes_df)
        train_diabetes_readmission_model(diabetes_df)
    
    if heart_df is not None:
        cardiovascular_params = extract_cardiovascular_parameters(heart_df)
        train_heart_disease_model(heart_df)
    
    # Save parameters
    all_params = save_parameters_to_json(metabolic_params, cardiovascular_params)
    
    # Summary
    print("\n" + "="*80)
    print("✅ REAL DATA INTEGRATION COMPLETE")
    print("="*80)
    
    print(f"""
📊 Data Processed:
  • Diabetes patients: {len(diabetes_df):,} (if loaded)
  • Heart disease patients: {len(heart_df):,} (if loaded)

🔬 Parameters Extracted:
  • Metabolic: {len(metabolic_params)} parameters
  • Cardiovascular: {len(cardiovascular_params)} parameters

🤖 Models Trained:
  • Diabetes readmission predictor
  • Heart disease risk predictor

📁 Files Created:
  • data/real/extracted_parameters.json
  • models/real_data/diabetes_readmission_model.pkl
  • models/real_data/heart_disease_model.pkl

🎯 Next Steps:
  1. Update organ_agents.py to use these real parameters
  2. Integrate models into hybrid_digital_twin.py
  3. Validate predictions against real outcomes
  4. Find additional datasets for other organs (renal, hepatic)

This replaces arbitrary parameters with REAL patient data! 🚀
""")


if __name__ == "__main__":
    main()
