#!/usr/bin/env python3
"""
Train machine learning models on the generated synthetic data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_features(df):
    """Prepare features for ML models"""
    
    # Select relevant features
    feature_cols = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'glucose_mmol_l', 'hba1c_percent', 'total_cholesterol_mmol_l',
        'ldl_cholesterol_mmol_l', 'hdl_cholesterol_mmol_l',
        'exercise_hours_per_week', 'sleep_hours_per_night',
        'alcohol_units_per_week', 'diet_quality_score', 'stress_level'
    ]
    
    X = df[feature_cols].copy()
    
    # Add gender encoding
    X['gender_male'] = (df['gender'] == 'male').astype(int)
    
    # Add smoking encoding
    X['smoking_current'] = (df['smoking_status'] == 'current').astype(int)
    X['smoking_former'] = (df['smoking_status'] == 'former').astype(int)
    
    return X


def train_diabetes_model(df):
    """Train diabetes prediction model"""
    logger.info("\n" + "="*80)
    logger.info("Training Diabetes Prediction Model")
    logger.info("="*80)
    
    X = prepare_features(df)
    y = df['diabetes'].astype(int)
    
    logger.info(f"Dataset: {len(X):,} patients")
    logger.info(f"Diabetes prevalence: {y.mean()*100:.1f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train):,} | Test set: {len(X_test):,}")
    
    # Train model
    logger.info("\nTraining Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("\n--- Model Performance ---")
    logger.info(f"Accuracy: {model.score(X_test, y_test):.3f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/diabetes_model.pkl")
    logger.info("\n✅ Model saved to models/diabetes_model.pkl")
    
    return model


def train_cvd_model(df):
    """Train cardiovascular disease prediction model"""
    logger.info("\n" + "="*80)
    logger.info("Training Cardiovascular Disease Prediction Model")
    logger.info("="*80)
    
    X = prepare_features(df)
    y = df['heart_disease'].astype(int)
    
    logger.info(f"Dataset: {len(X):,} patients")
    logger.info(f"CVD prevalence: {y.mean()*100:.1f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train):,} | Test set: {len(X_test):,}")
    
    logger.info("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("\n--- Model Performance ---")
    logger.info(f"Accuracy: {model.score(X_test, y_test):.3f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    joblib.dump(model, "models/cvd_model.pkl")
    logger.info("\n✅ Model saved to models/cvd_model.pkl")
    
    return model


def train_hypertension_model(df):
    """Train hypertension prediction model"""
    logger.info("\n" + "="*80)
    logger.info("Training Hypertension Prediction Model")
    logger.info("="*80)
    
    X = prepare_features(df)
    y = df['hypertension'].astype(int)
    
    logger.info(f"Dataset: {len(X):,} patients")
    logger.info(f"Hypertension prevalence: {y.mean()*100:.1f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("\n--- Model Performance ---")
    logger.info(f"Accuracy: {model.score(X_test, y_test):.3f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    
    joblib.dump(model, "models/hypertension_model.pkl")
    logger.info("\n✅ Model saved to models/hypertension_model.pkl")
    
    return model


if __name__ == "__main__":
    logger.info("Loading patient data...")
    df = pd.read_csv("data/synthetic/complete_patient_data.csv")
    
    # Train all models
    diabetes_model = train_diabetes_model(df)
    cvd_model = train_cvd_model(df)
    hypertension_model = train_hypertension_model(df)
    
    logger.info("\n" + "="*80)
    logger.info("✅ All models trained successfully!")
    logger.info("="*80)
    logger.info("\nModels saved in: models/")
    logger.info("  - diabetes_model.pkl")
    logger.info("  - cvd_model.pkl")
    logger.info("  - hypertension_model.pkl")
