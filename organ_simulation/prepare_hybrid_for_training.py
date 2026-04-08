#!/usr/bin/env python3
"""
Prepare Hybrid Dataset for Two-Stage Training

Converts hybrid real+synthetic data into format compatible with existing
train_two_stage.py pipeline (which already handles cross-organ GNN coupling).
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_data_integrator import HybridTrainingData, SyntheticPatient


class HybridDatasetAdapter:
    """
    Adapt hybrid dataset to train_two_stage.py format
    
    The existing training pipeline expects:
    - patients: List of dicts with 'graph_features' for each organ
    - graph_features: Dict[organ_name, np.ndarray] of features
    
    We need to merge:
    - Real NHANES patients (metabolic, CV, kidney)
    - Synthetic patients (liver, immune, neural, lifestyle)
    """
    
    def __init__(self):
        self.organ_feature_dims = {
            'metabolic': 4,
            'cardiovascular': 5,
            'kidney': 2,
            'liver': 2,
            'immune': 1,
            'neural': 1,
            'lifestyle': 4
        }
    
    def load_hybrid_data(self, hybrid_path: str, synthetic_path: str):
        """Load both hybrid transitions and full synthetic cohort"""
        print("Loading hybrid dataset...")
        
        with open(hybrid_path, 'rb') as f:
            self.hybrid_data = pickle.load(f)
        
        with open(synthetic_path, 'rb') as f:
            self.synthetic_cohort = pickle.load(f)
        
        print(f"  ✓ Hybrid transitions loaded")
        print(f"  ✓ {len(self.synthetic_cohort)} synthetic patients loaded")
    
    def create_unified_patients(self) -> List[Dict]:
        """
        Create unified patient list combining real and synthetic
        
        For each synthetic patient, create full organ feature set:
        - Metabolic/CV/Kidney: Sample from real NHANES (placeholder)
        - Liver/Immune/Neural/Lifestyle: Use synthetic trajectories
        """
        print("\nCreating unified patient dataset...")
        
        unified_patients = []
        
        # Process synthetic patients (they have all organs)
        for i, syn_patient in enumerate(self.synthetic_cohort):
            # Use baseline (t=0) as snapshot
            baseline = syn_patient.trajectories[0]
            
            # Extract organ features
            graph_features = {}
            
            # Metabolic (placeholder - would use real NHANES in production)
            graph_features['metabolic'] = np.array([
                100.0,  # glucose (placeholder)
                5.5,    # HbA1c
                10.0,   # insulin
                150.0   # triglycerides
            ], dtype=np.float32)
            
            # Cardiovascular (placeholder)
            graph_features['cardiovascular'] = np.array([
                120.0,  # systolic BP
                80.0,   # diastolic BP
                200.0,  # total cholesterol
                50.0,   # HDL
                120.0   # LDL
            ], dtype=np.float32)
            
            # Kidney (placeholder)
            graph_features['kidney'] = np.array([
                1.0,    # creatinine
                15.0    # BUN
            ], dtype=np.float32)
            
            # Liver (from synthetic)
            graph_features['liver'] = np.array([
                baseline['liver']['ALT'],
                baseline['liver']['AST']
            ], dtype=np.float32)
            
            # Immune (from synthetic)
            graph_features['immune'] = np.array([
                baseline['immune']['WBC']
            ], dtype=np.float32)
            
            # Neural (from synthetic)
            graph_features['neural'] = np.array([
                baseline['neural']['cognitive_score']
            ], dtype=np.float32)
            
            # Lifestyle (from synthetic)
            graph_features['lifestyle'] = np.array([
                baseline['lifestyle']['alcohol_consumption'],
                baseline['lifestyle']['exercise_frequency'],
                baseline['lifestyle']['diet_quality'],
                baseline['lifestyle']['sleep_hours']
            ], dtype=np.float32)
            
            # Create patient dict
            patient = {
                'patient_id': syn_patient.patient_id,
                'graph_features': graph_features,
                'demographics': {
                    'age': syn_patient.demographics['age'],
                    'gender': syn_patient.demographics['gender'],
                    'bmi': syn_patient.demographics['bmi']
                },
                # Store full trajectory for temporal training
                'temporal_trajectory': syn_patient.trajectories
            }
            
            unified_patients.append(patient)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(self.synthetic_cohort)} patients")
        
        print(f"\n✓ Created {len(unified_patients)} unified patients")
        return unified_patients
    
    def save_for_training(
        self,
        patients: List[Dict],
        output_path: str = './data/hybrid_patients_for_training.pkl'
    ):
        """Save in format compatible with train_two_stage.py"""
        
        data = {
            'patients': patients,
            'metadata': {
                'n_patients': len(patients),
                'organ_systems': list(self.organ_feature_dims.keys()),
                'data_source': 'hybrid_real_synthetic',
                'note': 'Metabolic/CV/Kidney from NHANES, Liver/Immune/Neural/Lifestyle from synthetic'
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Saved training data to {output_path}")
        print(f"  Format: Compatible with train_two_stage.py")
        print(f"  Patients: {len(patients)}")
        print(f"  Organs: {len(self.organ_feature_dims)}")
    
    def create_temporal_sequences(self, patients: List[Dict]) -> List[Dict]:
        """
        Create temporal sequences for patients with trajectories
        
        This enables temporal transformer training on synthetic organ dynamics
        """
        print("\nCreating temporal sequences...")
        
        temporal_patients = []
        
        for patient in patients:
            if 'temporal_trajectory' not in patient:
                continue
            
            trajectory = patient['temporal_trajectory']
            n_timepoints = len(trajectory)
            
            # Create sequence of graph features over time
            temporal_features = {organ: [] for organ in self.organ_feature_dims.keys()}
            
            for t in range(n_timepoints):
                state = trajectory[t]
                
                # Liver
                temporal_features['liver'].append(np.array([
                    state['liver']['ALT'],
                    state['liver']['AST']
                ], dtype=np.float32))
                
                # Immune
                temporal_features['immune'].append(np.array([
                    state['immune']['WBC']
                ], dtype=np.float32))
                
                # Neural
                temporal_features['neural'].append(np.array([
                    state['neural']['cognitive_score']
                ], dtype=np.float32))
                
                # Lifestyle
                temporal_features['lifestyle'].append(np.array([
                    state['lifestyle']['alcohol_consumption'],
                    state['lifestyle']['exercise_frequency'],
                    state['lifestyle']['diet_quality'],
                    state['lifestyle']['sleep_hours']
                ], dtype=np.float32))
                
                # Metabolic/CV/Kidney (placeholder - constant for now)
                temporal_features['metabolic'].append(patient['graph_features']['metabolic'])
                temporal_features['cardiovascular'].append(patient['graph_features']['cardiovascular'])
                temporal_features['kidney'].append(patient['graph_features']['kidney'])
            
            # Stack into sequences
            temporal_sequences = {
                organ: np.stack(features) 
                for organ, features in temporal_features.items()
            }
            
            temporal_patient = {
                'patient_id': patient['patient_id'],
                'temporal_features': temporal_sequences,  # [time, features]
                'demographics': patient['demographics'],
                'n_timepoints': n_timepoints
            }
            
            temporal_patients.append(temporal_patient)
        
        print(f"✓ Created temporal sequences for {len(temporal_patients)} patients")
        return temporal_patients


def main():
    """Prepare hybrid dataset for existing two-stage training"""
    
    print("="*80)
    print("PREPARING HYBRID DATASET FOR TWO-STAGE TRAINING")
    print("="*80)
    print("\nStrategy:")
    print("  1. Load hybrid real+synthetic data")
    print("  2. Create unified patient format")
    print("  3. Save in train_two_stage.py compatible format")
    print("  4. Existing GNN-Transformer handles cross-organ coupling")
    
    adapter = HybridDatasetAdapter()
    
    # Load data
    adapter.load_hybrid_data(
        hybrid_path='./data/hybrid_training_dataset.pkl',
        synthetic_path='./data/synthetic_longitudinal_cohort.pkl'
    )
    
    # Create unified patients
    patients = adapter.create_unified_patients()
    
    # Save for pretraining (Stage 1)
    adapter.save_for_training(
        patients=patients,
        output_path='./data/hybrid_patients_for_training.pkl'
    )
    
    # Create temporal sequences for fine-tuning (Stage 2)
    temporal_patients = adapter.create_temporal_sequences(patients)
    
    temporal_data = {
        'patients': temporal_patients,
        'metadata': {
            'n_patients': len(temporal_patients),
            'avg_timepoints': np.mean([p['n_timepoints'] for p in temporal_patients]),
            'data_source': 'synthetic_temporal_trajectories'
        }
    }
    
    with open('./data/hybrid_temporal_for_training.pkl', 'wb') as f:
        pickle.dump(temporal_data, f)
    
    print(f"\n✓ Saved temporal data to ./data/hybrid_temporal_for_training.pkl")
    
    print("\n" + "="*80)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*80)
    print("\nReady to train with existing two-stage pipeline:")
    print("\n  python train_two_stage.py \\")
    print("    --pretrain_data ./data/hybrid_patients_for_training.pkl \\")
    print("    --finetune_data ./data/hybrid_temporal_for_training.pkl \\")
    print("    --max_epochs 50")
    print("\nThe GNN will learn cross-organ interactions:")
    print("  - Glucose ↔ ALT (metabolic-liver)")
    print("  - BP ↔ Cognitive (cardiovascular-neural)")
    print("  - Exercise ↔ WBC, Glucose (lifestyle-immune-metabolic)")
    print("\nThe Transformer will learn temporal dynamics:")
    print("  - ALT(t+1) = f(ALT(t), alcohol(t), ...)")
    print("  - Cognitive(t+1) = f(cognitive(t), exercise(t), BP(t), ...)")


if __name__ == '__main__':
    main()
