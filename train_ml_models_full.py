#!/usr/bin/env python3
"""
Train ML models on the FULL 5 million patient dataset
Uses batch loading to handle large data efficiently
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_patient_data():
    """Load all patient data from batch files"""
    logger.info("Loading all patient data from batch files...")
    
    data_dir = Path("data/synthetic")
    batch_files = sorted(data_dir.glob("batch_*.csv"))
    
    logger.info(f"Found {len(batch_files)} batch files")
    
    all_data = []
    
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        df = pd.read_csv(batch_file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total patients loaded: {len(combined_df):,}")
    
    return combined_df


def prepare_features(df):
    """Prepare features for ML models"""
    feature_cols = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'glucose_mmol_l', 'hba1c_percent', 'total_cholesterol_mmol_l',
        'ldl_cholesterol_mmol_l', 'hdl_cholesterol_mmol_l',
        'exercise_hours_per_week', 'sleep_hours_per_night',
        'alcohol_units_per_week', 'diet_quality_score', 'stress_level'
    ]
    
    X = df[feature_cols].copy()
    X['gender_male'] = (df['gender'] == 'male').astype(int)
    X['smoking_current'] = (df['smoking_status'] == 'current').astype(int)
    X['smoking_former'] = (df['smoking_status'] == 'former').astype(int)
    
    return X


def train_large_scale_model(X, y, model_name, model_type='gradient_boosting'):
    """Train model on large dataset with progress tracking"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name} on {len(X):,} patients")
    logger.info(f"{'='*80}")
    
    logger.info(f"Positive class prevalence: {y.mean()*100:.1f}%")
    
    # Split data
    logger.info("Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train):,} | Test set: {len(X_test):,}")
    
    # Train model
    if model_type == 'gradient_boosting':
        logger.info("Training Gradient Boosting Classifier...")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=1
        )
    else:  # random_forest
        logger.info("Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    
    logger.info("Training in progress (this may take 10-30 minutes)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"\n{'='*80}")
    logger.info("MODEL PERFORMANCE")
    logger.info(f"{'='*80}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {cm[0,0]:,}")
    logger.info(f"  False Positives: {cm[0,1]:,}")
    logger.info(f"  False Negatives: {cm[1,0]:,}")
    logger.info(f"  True Positives:  {cm[1,1]:,}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return model, accuracy, roc_auc, feature_importance


def main():
    logger.info("="*80)
    logger.info("FULL-SCALE ML MODEL TRAINING (5 Million Patients)")
    logger.info("="*80)
    logger.info("\nThis will train models on ALL 5 million patients")
    logger.info("Estimated time: 30-60 minutes per model")
    logger.info("="*80)
    
    # Load all data
    df = load_all_patient_data()
    
    # Prepare features
    logger.info("\nPreparing features...")
    X = prepare_features(df)
    
    # Create models directory
    Path("models/full_scale").mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Train Diabetes Model
    logger.info("\n" + "="*80)
    logger.info("1/3: DIABETES PREDICTION MODEL")
    logger.info("="*80)
    y_diabetes = df['diabetes'].astype(int)
    diabetes_model, diabetes_acc, diabetes_auc, diabetes_fi = train_large_scale_model(
        X, y_diabetes, "Diabetes Prediction", model_type='gradient_boosting'
    )
    joblib.dump(diabetes_model, "models/full_scale/diabetes_model_5M.pkl")
    logger.info("\n✅ Diabetes model saved to models/full_scale/diabetes_model_5M.pkl")
    results['diabetes'] = {'accuracy': diabetes_acc, 'roc_auc': diabetes_auc}
    
    # Train CVD Model
    logger.info("\n" + "="*80)
    logger.info("2/3: CARDIOVASCULAR DISEASE PREDICTION MODEL")
    logger.info("="*80)
    y_cvd = df['heart_disease'].astype(int)
    cvd_model, cvd_acc, cvd_auc, cvd_fi = train_large_scale_model(
        X, y_cvd, "CVD Prediction", model_type='random_forest'
    )
    joblib.dump(cvd_model, "models/full_scale/cvd_model_5M.pkl")
    logger.info("\n✅ CVD model saved to models/full_scale/cvd_model_5M.pkl")
    results['cvd'] = {'accuracy': cvd_acc, 'roc_auc': cvd_auc}
    
    # Train Hypertension Model
    logger.info("\n" + "="*80)
    logger.info("3/3: HYPERTENSION PREDICTION MODEL")
    logger.info("="*80)
    y_hypertension = df['hypertension'].astype(int)
    hypertension_model, hypertension_acc, hypertension_auc, hypertension_fi = train_large_scale_model(
        X, y_hypertension, "Hypertension Prediction", model_type='gradient_boosting'
    )
    joblib.dump(hypertension_model, "models/full_scale/hypertension_model_5M.pkl")
    logger.info("\n✅ Hypertension model saved to models/full_scale/hypertension_model_5M.pkl")
    results['hypertension'] = {'accuracy': hypertension_acc, 'roc_auc': hypertension_auc}
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*80)
    logger.info(f"\nDataset size: {len(df):,} patients")
    logger.info("\nModel Performance:")
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ All full-scale models trained successfully!")
    logger.info("="*80)
    logger.info("\nModels saved in: models/full_scale/")
    logger.info("  - diabetes_model_5M.pkl")
    logger.info("  - cvd_model_5M.pkl")
    logger.info("  - hypertension_model_5M.pkl")
    
    # Save feature importance
    diabetes_fi.to_csv("models/full_scale/diabetes_feature_importance.csv", index=False)
    cvd_fi.to_csv("models/full_scale/cvd_feature_importance.csv", index=False)
    hypertension_fi.to_csv("models/full_scale/hypertension_feature_importance.csv", index=False)
    logger.info("\nFeature importance saved to CSV files")


if __name__ == "__main__":
    main()
