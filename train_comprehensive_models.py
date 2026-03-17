#!/usr/bin/env python3
"""
Train ML models on 108k patient dataset for all 7 agents
Replaces medical theory estimates with data-driven predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveModelTrainer:
    """Train models for all 7 organ agents"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = {}
        self.results = {}
        
    def load_diabetes_data(self):
        """Load diabetes dataset (101,766 patients)"""
        print("\n📊 Loading diabetes dataset...")
        df = pd.read_csv('data/real/raw/dataset_diabetes/diabetic_data.csv')
        print(f"   Loaded {len(df):,} diabetes patients")
        return df
    
    def load_heart_disease_data(self):
        """Load heart disease dataset"""
        print("\n📊 Loading heart disease dataset...")
        # Use UCI downloaded version - add header names
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        df = pd.read_csv('data/downloaded_datasets/uci/uci_heart_disease_cleveland.csv', 
                        names=column_names, na_values='?')
        print(f"   Loaded {len(df):,} heart disease patients")
        return df
    
    def load_liver_data(self):
        """Load liver disease dataset"""
        print("\n📊 Loading liver disease dataset...")
        try:
            df = pd.read_csv('data/real/raw/liver_disease.csv')
            print(f"   Loaded {len(df):,} liver patients")
            return df
        except:
            print("   ⚠️  Liver data not found, skipping")
            return None
    
    def train_metabolic_agent(self):
        """Train diabetes prediction model"""
        print("\n" + "="*80)
        print("TRAINING METABOLIC AGENT (Diabetes)")
        print("="*80)
        
        df = self.load_diabetes_data()
        
        # Prepare features
        age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                   '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                   '[80-90)': 85, '[90-100)': 95}
        
        df['age_numeric'] = df['age'].map(age_map)
        df['gender_numeric'] = (df['gender'] == 'Male').astype(int)
        
        # Target: readmission within 30 days
        df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
        
        # Features
        feature_cols = ['age_numeric', 'gender_numeric', 'time_in_hospital',
                       'num_lab_procedures', 'num_procedures', 'num_medications',
                       'number_outpatient', 'number_emergency', 'number_inpatient',
                       'number_diagnoses']
        
        # Add medication features
        med_cols = ['metformin', 'glipizide', 'glyburide', 'insulin']
        for col in med_cols:
            if col in df.columns:
                df[f'{col}_changed'] = (df[col] != 'No').astype(int)
                feature_cols.append(f'{col}_changed')
        
        # Clean data
        df_clean = df[feature_cols + ['readmitted_30']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['readmitted_30']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Training on {len(X_train):,} patients...")
        print(f"   Positive cases: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        
        # Train model
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
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n✅ Model Performance:")
        print(f"   Accuracy: {accuracy*100:.1f}%")
        print(f"   AUC-ROC:  {auc:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 5 Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']:25} {row['importance']:.3f}")
        
        # Save model
        self.models['metabolic'] = model
        self.feature_columns['metabolic'] = feature_cols
        self.results['metabolic'] = {
            'accuracy': accuracy,
            'auc': auc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        return model
    
    def train_cardiovascular_agent(self):
        """Train heart disease prediction model"""
        print("\n" + "="*80)
        print("TRAINING CARDIOVASCULAR AGENT")
        print("="*80)
        
        df = self.load_heart_disease_data()
        
        # Check available columns
        print(f"   Columns: {df.columns.tolist()}")
        
        # Use 'num' as target (0 = no disease, 1-4 = disease)
        if 'num' in df.columns:
            df['target'] = (df['num'] > 0).astype(int)
        elif 'condition' in df.columns:
            df['target'] = df['condition']
        else:
            # Use last column as target
            df['target'] = (df.iloc[:, -1] > 0).astype(int)
        
        # Features - use available columns
        available_cols = df.columns.tolist()
        feature_cols = [col for col in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                       'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                       'ca', 'thal'] if col in available_cols]
        
        # Clean data
        df_clean = df[feature_cols + ['target']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Training on {len(X_train):,} patients...")
        print(f"   Heart disease cases: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n✅ Model Performance:")
        print(f"   Accuracy: {accuracy*100:.1f}%")
        print(f"   AUC-ROC:  {auc:.3f}")
        
        # Save model
        self.models['cardiovascular'] = model
        self.feature_columns['cardiovascular'] = feature_cols
        self.results['cardiovascular'] = {
            'accuracy': accuracy,
            'auc': auc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        return model
    
    def train_hepatic_agent(self):
        """Train liver disease prediction model"""
        print("\n" + "="*80)
        print("TRAINING HEPATIC AGENT")
        print("="*80)
        
        df = self.load_liver_data()
        
        if df is None:
            print("   ⚠️  Skipping hepatic agent (no data)")
            return None
        
        # Features
        feature_cols = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                       'Aspartate_Aminotransferase', 'Total_Protiens',
                       'Albumin', 'Albumin_and_Globulin_Ratio']
        
        # Encode gender
        df['Gender'] = (df['Gender'] == 'Male').astype(int)
        
        # Target
        df['Dataset'] = df['Dataset'] - 1  # Convert to 0/1
        
        # Clean data
        df_clean = df[feature_cols + ['Dataset']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['Dataset']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Training on {len(X_train):,} patients...")
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n✅ Model Performance:")
        print(f"   Accuracy: {accuracy*100:.1f}%")
        print(f"   AUC-ROC:  {auc:.3f}")
        
        # Save model
        self.models['hepatic'] = model
        self.feature_columns['hepatic'] = feature_cols
        self.results['hepatic'] = {
            'accuracy': accuracy,
            'auc': auc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        return model
    
    def save_models(self, output_dir='models/trained'):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("SAVING TRAINED MODELS")
        print("="*80)
        
        for agent_name, model in self.models.items():
            model_file = output_path / f'{agent_name}_model.pkl'
            
            model_data = {
                'model': model,
                'feature_columns': self.feature_columns[agent_name],
                'results': self.results[agent_name]
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"   ✅ Saved {agent_name} model to {model_file}")
        
        # Save summary
        summary_file = output_path / 'training_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE MODEL TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for agent_name, results in self.results.items():
                f.write(f"{agent_name.upper()} AGENT:\n")
                f.write(f"  Accuracy: {results['accuracy']*100:.1f}%\n")
                f.write(f"  AUC-ROC:  {results['auc']:.3f}\n")
                f.write(f"  Training samples: {results['n_train']:,}\n")
                f.write(f"  Test samples:     {results['n_test']:,}\n")
                f.write("\n")
        
        print(f"\n   📄 Saved training summary to {summary_file}")
        
        return output_path
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*80)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*80)
        
        total_patients = sum(r['n_train'] + r['n_test'] for r in self.results.values())
        
        print(f"\n📊 Total patients used: {total_patients:,}")
        print(f"   Agents trained: {len(self.models)}")
        
        print("\n🎯 Model Performance:")
        for agent_name, results in self.results.items():
            print(f"\n   {agent_name.upper()}:")
            print(f"      Accuracy: {results['accuracy']*100:.1f}%")
            print(f"      AUC-ROC:  {results['auc']:.3f}")
            print(f"      Samples:  {results['n_train'] + results['n_test']:,}")


def main():
    """Train all models"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL TRAINING")
    print("Training ML models on 108k patient dataset")
    print("="*80)
    
    trainer = ComprehensiveModelTrainer()
    
    # Train all agents
    trainer.train_metabolic_agent()
    trainer.train_cardiovascular_agent()
    # Skip hepatic for now - data format issues
    # trainer.train_hepatic_agent()
    
    # Save models
    trainer.save_models()
    
    # Print summary
    trainer.print_summary()
    
    print("\n✅ Models trained and saved!")
    print("   Models can now be loaded by the digital twin system")


if __name__ == "__main__":
    main()
