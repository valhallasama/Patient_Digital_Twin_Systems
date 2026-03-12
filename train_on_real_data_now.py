#!/usr/bin/env python3
"""
Train ML Models on Real Data (100K+ patients)
Uses the downloaded UCI Diabetes dataset
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pickle
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_diabetes_130_dataset():
    """Load the Diabetes 130-US Hospitals dataset"""
    
    logger.info("\n" + "="*80)
    logger.info("LOADING DIABETES 130-US HOSPITALS DATASET")
    logger.info("="*80)
    
    data_path = Path("data/real/raw/dataset_diabetes/diabetic_data.csv")
    
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"\n✓ Loaded {len(df):,} patient records")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    return df


def prepare_features(df):
    """Prepare features for ML training"""
    
    logger.info("\n" + "="*80)
    logger.info("PREPARING FEATURES")
    logger.info("="*80)
    
    # Key columns for diabetes prediction
    feature_cols = [
        'age', 'time_in_hospital', 'num_lab_procedures', 
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'number_diagnoses'
    ]
    
    # Target: readmitted (proxy for diabetes control)
    # We'll create a binary target: readmitted within 30 days = poor control
    
    # Map age ranges to numeric
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    
    df_clean = df.copy()
    df_clean['age_numeric'] = df_clean['age'].map(age_map)
    
    # Select features
    X_cols = ['age_numeric'] + feature_cols[1:]
    X = df_clean[X_cols].copy()
    
    # Create target: readmitted < 30 days = 1 (poor control)
    y = (df_clean['readmitted'] == '<30').astype(int)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    logger.info(f"\n✓ Features prepared")
    logger.info(f"  Feature columns: {len(X.columns)}")
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Positive cases (readmitted <30 days): {y.sum():,} ({y.mean():.1%})")
    
    return X, y


def train_model(X, y):
    """Train Gradient Boosting model"""
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING GRADIENT BOOSTING MODEL")
    logger.info("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"\nTrain set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    
    # Train model
    logger.info("\nTraining Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE")
    logger.info("="*80)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"\nAccuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    models_dir = Path("models/real_data")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "diabetes_readmission_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"\n✓ Model saved: {model_path}")
    
    return model, accuracy, roc_auc


def main():
    logger.info("\n" + "="*80)
    logger.info("TRAINING ON REAL DATA - 100K+ PATIENTS")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_diabetes_130_dataset()
    if df is None:
        logger.error("\n✗ Failed to load dataset")
        return
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    model, accuracy, roc_auc = train_model(X, y)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\n✓ Trained on {len(X):,} real patient records")
    logger.info(f"✓ Accuracy: {accuracy:.4f}")
    logger.info(f"✓ ROC-AUC: {roc_auc:.4f}")
    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("\n1. Compare with synthetic model:")
    logger.info("   python3 compare_real_vs_synthetic.py")
    logger.info("\n2. Use model for predictions:")
    logger.info("   import pickle")
    logger.info("   model = pickle.load(open('models/real_data/diabetes_readmission_model.pkl', 'rb'))")


if __name__ == "__main__":
    main()
