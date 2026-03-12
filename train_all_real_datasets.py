#!/usr/bin/env python3
"""
Train ML Models on ALL Real Datasets
- Diabetes 130-US Hospitals (101K patients)
- Heart Disease Cleveland (303 patients)
- Heart Disease Hungarian (294 patients)
Total: 102K+ patients across multiple datasets
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import pickle
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MultiDatasetTrainer:
    """Train ML models on multiple real datasets"""
    
    def __init__(self):
        self.models_dir = Path("models/real_data")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def load_diabetes_130(self):
        """Load Diabetes 130-US Hospitals dataset"""
        logger.info("\n" + "="*80)
        logger.info("DATASET 1: DIABETES 130-US HOSPITALS")
        logger.info("="*80)
        
        data_path = Path("data/real/raw/dataset_diabetes/diabetic_data.csv")
        df = pd.read_csv(data_path)
        
        logger.info(f"✓ Loaded {len(df):,} patient records")
        
        # Map age to numeric
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df['age_numeric'] = df['age'].map(age_map)
        
        # Features
        X = df[[
            'age_numeric', 'time_in_hospital', 'num_lab_procedures',
            'num_procedures', 'num_medications', 'number_outpatient',
            'number_emergency', 'number_inpatient', 'number_diagnoses'
        ]].fillna(0)
        
        # Target: readmitted < 30 days
        y = (df['readmitted'] == '<30').astype(int)
        
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Positive rate: {y.mean():.1%}")
        
        return X, y, 'diabetes_readmission'
    
    def load_heart_disease_cleveland(self):
        """Load Heart Disease Cleveland dataset"""
        logger.info("\n" + "="*80)
        logger.info("DATASET 2: HEART DISEASE CLEVELAND")
        logger.info("="*80)
        
        data_path = Path("data/real/raw/heart_disease_uci.csv")
        
        # Column names for Cleveland dataset
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        df = pd.read_csv(data_path, names=columns, na_values='?')
        
        logger.info(f"✓ Loaded {len(df):,} patient records")
        
        # Features (drop target)
        X = df.drop('target', axis=1)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Target: 0 = no disease, 1-4 = disease present
        y = (df['target'] > 0).astype(int)
        
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Positive rate: {y.mean():.1%}")
        
        return X, y, 'heart_disease_cleveland'
    
    def load_heart_disease_hungarian(self):
        """Load Heart Disease Hungarian dataset"""
        logger.info("\n" + "="*80)
        logger.info("DATASET 3: HEART DISEASE HUNGARIAN")
        logger.info("="*80)
        
        data_path = Path("data/real/raw/heart_disease_hungarian.csv")
        
        # Same columns as Cleveland
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        df = pd.read_csv(data_path, names=columns, na_values='?')
        
        logger.info(f"✓ Loaded {len(df):,} patient records")
        
        # Features
        X = df.drop('target', axis=1)
        X = X.fillna(X.median())
        
        # Target
        y = (df['target'] > 0).astype(int)
        
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Positive rate: {y.mean():.1%}")
        
        return X, y, 'heart_disease_hungarian'
    
    def train_model(self, X, y, dataset_name, model_type='gradient_boosting'):
        """Train a model on given dataset"""
        logger.info("\n" + "-"*80)
        logger.info(f"TRAINING: {dataset_name.upper()}")
        logger.info("-"*80)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Train model
        if model_type == 'gradient_boosting':
            logger.info("Model: Gradient Boosting Classifier")
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )
        else:
            logger.info("Model: Random Forest Classifier")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                verbose=0
            )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"\n✓ Accuracy: {accuracy:.4f}")
        logger.info(f"✓ ROC-AUC: {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        logger.info(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
        
        # Save model
        model_path = self.models_dir / f"{dataset_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"\n✓ Saved: {model_path}")
        
        # Store results
        self.results[dataset_name] = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'samples': len(X),
            'features': X.shape[1],
            'positive_rate': float(y.mean()),
            'model_path': str(model_path)
        }
        
        return model, accuracy, roc_auc
    
    def train_all(self):
        """Train models on all datasets"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING ON ALL REAL DATASETS")
        logger.info("="*80)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Dataset 1: Diabetes 130-US
        X1, y1, name1 = self.load_diabetes_130()
        model1, acc1, auc1 = self.train_model(X1, y1, name1)
        
        # Dataset 2: Heart Disease Cleveland
        X2, y2, name2 = self.load_heart_disease_cleveland()
        model2, acc2, auc2 = self.train_model(X2, y2, name2)
        
        # Dataset 3: Heart Disease Hungarian
        X3, y3, name3 = self.load_heart_disease_hungarian()
        model3, acc3, auc3 = self.train_model(X3, y3, name3)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        
        total_samples = sum(r['samples'] for r in self.results.values())
        avg_accuracy = np.mean([r['accuracy'] for r in self.results.values()])
        avg_roc_auc = np.mean([r['roc_auc'] for r in self.results.values()])
        
        logger.info(f"\nTotal datasets: {len(self.results)}")
        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"Average accuracy: {avg_accuracy:.4f}")
        logger.info(f"Average ROC-AUC: {avg_roc_auc:.4f}")
        
        logger.info("\nPer-Dataset Performance:")
        for name, result in self.results.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Samples: {result['samples']:,}")
            logger.info(f"  Accuracy: {result['accuracy']:.4f}")
            logger.info(f"  ROC-AUC: {result['roc_auc']:.4f}")
        
        # Save summary
        summary_path = self.models_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'trained_at': datetime.now().isoformat(),
                'total_samples': total_samples,
                'total_datasets': len(self.results),
                'average_accuracy': float(avg_accuracy),
                'average_roc_auc': float(avg_roc_auc),
                'datasets': self.results
            }, f, indent=2)
        
        logger.info(f"\n✓ Summary saved: {summary_path}")
        
        logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results


def main():
    logger.info("\n" + "="*80)
    logger.info("MULTI-DATASET ML TRAINING")
    logger.info("="*80)
    logger.info("Training on ALL available real datasets")
    logger.info("This will take 2-5 minutes...")
    
    trainer = MultiDatasetTrainer()
    results = trainer.train_all()
    
    logger.info("\n" + "="*80)
    logger.info("ALL TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("\n✓ Trained 3 models on 102K+ real patients")
    logger.info("✓ All models saved to models/real_data/")
    logger.info("\nNext steps:")
    logger.info("  1. Compare with synthetic: python3 compare_real_vs_synthetic.py")
    logger.info("  2. Use models for predictions")


if __name__ == "__main__":
    main()
