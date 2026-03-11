"""
ML Training Pipeline for Real Data
Trains models on real patient datasets (not synthetic)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import pickle
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple

from data_engine.real_data_pipeline import RealDataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataMLTrainer:
    """
    Train ML models on real patient data
    """
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path("models/real_data")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipeline = RealDataPipeline()
        
        logger.info("✓ Real Data ML Trainer initialized")
        logger.info(f"  Models directory: {self.models_dir}")
    
    def load_all_cleaned_datasets(self) -> pd.DataFrame:
        """
        Load and combine all cleaned real datasets
        """
        logger.info("\n" + "="*80)
        logger.info("LOADING REAL DATASETS")
        logger.info("="*80)
        
        # Get all cleaned datasets
        datasets = self.pipeline.lineage.get_all_datasets(validated_only=True)
        cleaned_datasets = [d for d in datasets if d.get('cleaned', False)]
        
        logger.info(f"Found {len(cleaned_datasets)} cleaned datasets")
        
        if not cleaned_datasets:
            logger.error("No cleaned datasets available!")
            logger.error("Run data acquisition first: python3 run_daily_data_acquisition.py")
            return None
        
        # Load and combine
        dfs = []
        total_rows = 0
        
        for i, dataset in enumerate(cleaned_datasets, 1):
            try:
                cleaned_path = Path(dataset['cleaned_path'])
                
                if not cleaned_path.exists():
                    logger.warning(f"  [{i}] File not found: {cleaned_path}")
                    continue
                
                df = pd.read_csv(cleaned_path)
                dfs.append(df)
                total_rows += len(df)
                
                logger.info(f"  [{i}] Loaded: {dataset['title'][:50]}... ({len(df)} rows)")
                
            except Exception as e:
                logger.error(f"  [{i}] Error loading {dataset['dataset_id']}: {e}")
        
        if not dfs:
            logger.error("No datasets could be loaded!")
            return None
        
        # Combine all datasets
        combined = pd.concat(dfs, ignore_index=True)
        
        logger.info("\n" + "="*80)
        logger.info(f"COMBINED DATASET: {len(combined)} total rows from {len(dfs)} datasets")
        logger.info("="*80)
        
        return combined
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare features for ML training
        Handles different dataset formats
        """
        logger.info("\nPreparing features...")
        
        # Common health-related column names (case-insensitive)
        feature_mappings = {
            'age': ['age', 'Age', 'AGE', 'patient_age', 'PatientAge'],
            'gender': ['gender', 'Gender', 'sex', 'Sex', 'SEX'],
            'bmi': ['bmi', 'BMI', 'body_mass_index', 'BodyMassIndex'],
            'glucose': ['glucose', 'Glucose', 'blood_glucose', 'fasting_glucose', 'FBG'],
            'hba1c': ['hba1c', 'HbA1c', 'A1C', 'glycated_hemoglobin'],
            'systolic_bp': ['systolic', 'SBP', 'systolic_bp', 'systolic_blood_pressure'],
            'diastolic_bp': ['diastolic', 'DBP', 'diastolic_bp', 'diastolic_blood_pressure'],
            'cholesterol': ['cholesterol', 'total_cholesterol', 'chol', 'CHOL'],
            'ldl': ['ldl', 'LDL', 'ldl_cholesterol', 'LDL_cholesterol'],
            'hdl': ['hdl', 'HDL', 'hdl_cholesterol', 'HDL_cholesterol'],
            'triglycerides': ['triglycerides', 'trig', 'TG'],
            'smoking': ['smoking', 'smoker', 'smoking_status', 'tobacco'],
            'diabetes': ['diabetes', 'diabetic', 'has_diabetes', 'diabetes_diagnosis'],
            'cvd': ['cvd', 'heart_disease', 'cardiovascular_disease', 'CHD'],
            'hypertension': ['hypertension', 'high_blood_pressure', 'HTN']
        }
        
        # Find matching columns
        standardized_df = pd.DataFrame()
        found_columns = {}
        
        for standard_name, possible_names in feature_mappings.items():
            for col in df.columns:
                if col in possible_names:
                    standardized_df[standard_name] = df[col]
                    found_columns[standard_name] = col
                    break
        
        logger.info(f"  Found {len(found_columns)} standard features:")
        for std_name, orig_name in found_columns.items():
            logger.info(f"    {std_name} <- {orig_name}")
        
        # Handle missing values
        for col in standardized_df.columns:
            if standardized_df[col].dtype in [np.float64, np.int64]:
                # Numeric: fill with median
                standardized_df[col].fillna(standardized_df[col].median(), inplace=True)
            else:
                # Categorical: fill with mode
                standardized_df[col].fillna(standardized_df[col].mode()[0] if len(standardized_df[col].mode()) > 0 else 'unknown', inplace=True)
        
        # Encode categorical variables
        if 'gender' in standardized_df.columns:
            standardized_df['gender'] = standardized_df['gender'].map({'male': 1, 'Male': 1, 'M': 1, 'female': 0, 'Female': 0, 'F': 0}).fillna(0)
        
        if 'smoking' in standardized_df.columns:
            standardized_df['smoking'] = standardized_df['smoking'].map({'yes': 1, 'Yes': 1, 'current': 1, 'no': 0, 'No': 0, 'never': 0}).fillna(0)
        
        logger.info(f"  Final feature set: {list(standardized_df.columns)}")
        logger.info(f"  Shape: {standardized_df.shape}")
        
        return standardized_df, found_columns
    
    def train_disease_model(self, df: pd.DataFrame, disease: str, 
                           target_col: str) -> Dict:
        """
        Train model for specific disease
        """
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING MODEL: {disease.upper()}")
        logger.info("="*80)
        
        # Check if target exists
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found!")
            logger.error(f"Available columns: {list(df.columns)}")
            return None
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Convert target to binary if needed
        if y.dtype == 'object':
            y = y.map({'yes': 1, 'Yes': 1, 'true': 1, 'True': 1, 1: 1, 'no': 0, 'No': 0, 'false': 0, 'False': 0, 0: 0}).fillna(0)
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Positive cases: {y.sum()} ({y.mean():.1%})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"\nTraining set: {len(X_train)}")
        logger.info(f"Test set: {len(X_test)}")
        
        # Train Gradient Boosting
        logger.info("\nTraining Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
        
        gb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = gb_model.predict(X_test)
        y_pred_proba = gb_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("\n" + "="*80)
        logger.info("MODEL PERFORMANCE")
        logger.info("="*80)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        logger.info("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        model_path = self.models_dir / f"{disease}_model_real.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(gb_model, f)
        
        logger.info(f"\n✓ Model saved: {model_path}")
        
        # Save metadata
        metadata = {
            'disease': disease,
            'trained_at': datetime.now().isoformat(),
            'data_source': 'real_datasets',
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'features': feature_cols,
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'positive_rate': float(y.mean()),
            'feature_importance': feature_importance.to_dict('records')
        }
        
        metadata_path = self.models_dir / f"{disease}_metadata_real.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved: {metadata_path}")
        
        return {
            'model': gb_model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'metadata': metadata
        }
    
    def train_all_models(self) -> Dict:
        """
        Train models for all diseases using real data
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING ALL MODELS ON REAL DATA")
        logger.info("="*80)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load all real datasets
        df = self.load_all_cleaned_datasets()
        
        if df is None or len(df) == 0:
            logger.error("No data available for training!")
            return None
        
        # Prepare features
        df_prepared, found_columns = self.prepare_features(df)
        
        # Train models for each disease
        results = {}
        
        # Diabetes
        if 'diabetes' in df_prepared.columns:
            results['diabetes'] = self.train_disease_model(
                df_prepared, 'diabetes', 'diabetes'
            )
        
        # CVD
        if 'cvd' in df_prepared.columns:
            results['cvd'] = self.train_disease_model(
                df_prepared, 'cvd', 'cvd'
            )
        
        # Hypertension
        if 'hypertension' in df_prepared.columns:
            results['hypertension'] = self.train_disease_model(
                df_prepared, 'hypertension', 'hypertension'
            )
        
        logger.info("\n" + "="*80)
        logger.info("ALL MODELS TRAINED")
        logger.info("="*80)
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Models trained: {len(results)}")
        
        for disease, result in results.items():
            if result:
                logger.info(f"  {disease}: ROC-AUC = {result['roc_auc']:.4f}")
        
        return results


# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ML TRAINING ON REAL DATA")
    print("="*80)
    
    # Initialize trainer
    trainer = RealDataMLTrainer()
    
    # Train all models
    results = trainer.train_all_models()
    
    if results:
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)
        print(f"\nModels saved in: {trainer.models_dir}")
        print("\nModel Performance:")
        for disease, result in results.items():
            if result:
                print(f"  {disease.upper()}:")
                print(f"    Accuracy: {result['accuracy']:.4f}")
                print(f"    ROC-AUC: {result['roc_auc']:.4f}")
    else:
        print("\n" + "="*80)
        print("✗ TRAINING FAILED")
        print("="*80)
        print("\nPossible reasons:")
        print("  1. No real datasets downloaded yet")
        print("  2. Datasets don't contain required columns")
        print("\nSolutions:")
        print("  1. Run data acquisition: python3 run_daily_data_acquisition.py")
        print("  2. Wait for datasets to download and validate")
        print("  3. Check data/real/lineage.json for available datasets")
