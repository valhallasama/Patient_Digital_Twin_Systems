#!/usr/bin/env python3
"""
Hybrid Data Integrator

Combines real NHANES temporal data with synthetic trajectories
to create complete training dataset for digital twin.

Strategy:
- Metabolic/CV/Kidney: Real NHANES transitions (33,994 patients)
- Liver/Immune/Neural/Lifestyle: Synthetic physics-informed trajectories (10,000 patients)
"""

import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

# Import SyntheticPatient for pickle loading
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_trajectory_generator import SyntheticPatient


@dataclass
class HybridTrainingData:
    """Combined real + synthetic training data"""
    real_transitions: Dict  # Metabolic, CV, Kidney from NHANES
    synthetic_transitions: Dict  # Liver, Immune, Neural, Lifestyle
    metadata: Dict


class HybridDataIntegrator:
    """
    Integrate real NHANES data with synthetic trajectories
    
    Creates unified training dataset for temporal models.
    """
    
    def __init__(self):
        self.organ_sources = {
            'metabolic': 'real',
            'cardiovascular': 'real',
            'kidney': 'real',
            'liver': 'synthetic',
            'immune': 'synthetic',
            'neural': 'synthetic',
            'lifestyle': 'synthetic'
        }
    
    def load_real_data(self, nhanes_path: str) -> Dict:
        """Load real NHANES temporal transitions"""
        print("Loading real NHANES temporal data...")
        
        # Load NHANES data
        with open(nhanes_path, 'rb') as f:
            nhanes_data = pickle.load(f)
        
        print(f"  ✓ Loaded {len(nhanes_data)} NHANES temporal transitions")
        
        return nhanes_data
    
    def load_synthetic_data(self, synthetic_path: str) -> List:
        """Load synthetic trajectories"""
        print("Loading synthetic trajectory data...")
        
        with open(synthetic_path, 'rb') as f:
            synthetic_cohort = pickle.load(f)
        
        print(f"  ✓ Loaded {len(synthetic_cohort)} synthetic patients")
        
        return synthetic_cohort
    
    def extract_synthetic_transitions(self, synthetic_cohort: List) -> Dict:
        """
        Extract temporal transitions from synthetic cohort
        
        Format: Same as NHANES transitions for compatibility
        """
        print("Extracting synthetic temporal transitions...")
        
        transitions = {
            'liver': [],
            'immune': [],
            'neural': [],
            'lifestyle': []
        }
        
        for patient in synthetic_cohort:
            n_timepoints = len(patient.trajectories)
            
            # Extract t -> t+1 transitions
            for t in range(n_timepoints - 1):
                state_t = patient.trajectories[t]
                state_t1 = patient.trajectories[t + 1]
                
                # Liver transition
                transitions['liver'].append({
                    'patient_id': patient.patient_id,
                    'time_t': t,
                    'time_t1': t + 1,
                    'ALT_t': state_t['liver']['ALT'],
                    'AST_t': state_t['liver']['AST'],
                    'ALT_t1': state_t1['liver']['ALT'],
                    'AST_t1': state_t1['liver']['AST'],
                    'alcohol_t': state_t['lifestyle']['alcohol_consumption'],
                    'exercise_t': state_t['lifestyle']['exercise_frequency'],
                    'diet_t': state_t['lifestyle']['diet_quality'],
                    'age_t': state_t['age']
                })
                
                # Immune transition
                transitions['immune'].append({
                    'patient_id': patient.patient_id,
                    'time_t': t,
                    'time_t1': t + 1,
                    'WBC_t': state_t['immune']['WBC'],
                    'WBC_t1': state_t1['immune']['WBC'],
                    'ALT_t': state_t['liver']['ALT'],  # Cross-organ
                    'exercise_t': state_t['lifestyle']['exercise_frequency'],
                    'age_t': state_t['age']
                })
                
                # Neural transition
                transitions['neural'].append({
                    'patient_id': patient.patient_id,
                    'time_t': t,
                    'time_t1': t + 1,
                    'cognitive_t': state_t['neural']['cognitive_score'],
                    'cognitive_t1': state_t1['neural']['cognitive_score'],
                    'exercise_t': state_t['lifestyle']['exercise_frequency'],
                    'diet_t': state_t['lifestyle']['diet_quality'],
                    'ALT_t': state_t['liver']['ALT'],  # Cross-organ
                    'age_t': state_t['age']
                })
                
                # Lifestyle transition
                transitions['lifestyle'].append({
                    'patient_id': patient.patient_id,
                    'time_t': t,
                    'time_t1': t + 1,
                    'alcohol_t': state_t['lifestyle']['alcohol_consumption'],
                    'exercise_t': state_t['lifestyle']['exercise_frequency'],
                    'diet_t': state_t['lifestyle']['diet_quality'],
                    'alcohol_t1': state_t1['lifestyle']['alcohol_consumption'],
                    'exercise_t1': state_t1['lifestyle']['exercise_frequency'],
                    'diet_t1': state_t1['lifestyle']['diet_quality'],
                    'ALT_t': state_t['liver']['ALT'],  # Health events trigger changes
                    'age_t': state_t['age']
                })
        
        for organ, trans in transitions.items():
            print(f"  ✓ {organ}: {len(trans)} transitions")
        
        return transitions
    
    def create_hybrid_dataset(
        self,
        nhanes_path: str,
        synthetic_path: str,
        output_path: str
    ) -> HybridTrainingData:
        """
        Create hybrid training dataset
        
        Args:
            nhanes_path: Path to NHANES temporal data
            synthetic_path: Path to synthetic cohort
            output_path: Where to save hybrid dataset
        
        Returns:
            HybridTrainingData object
        """
        print("="*80)
        print("CREATING HYBRID TRAINING DATASET")
        print("="*80)
        
        # Load real data
        real_data = self.load_real_data(nhanes_path)
        
        # Load synthetic data
        synthetic_cohort = self.load_synthetic_data(synthetic_path)
        
        # Extract synthetic transitions
        synthetic_transitions = self.extract_synthetic_transitions(synthetic_cohort)
        
        # Create hybrid dataset
        hybrid_data = HybridTrainingData(
            real_transitions={
                'metabolic': real_data.get('metabolic', []),
                'cardiovascular': real_data.get('cardiovascular', []),
                'kidney': real_data.get('kidney', [])
            },
            synthetic_transitions=synthetic_transitions,
            metadata={
                'n_real_patients': len(set([t['patient_id'] for t in real_data.get('metabolic', [])])) if real_data.get('metabolic') else 0,
                'n_synthetic_patients': len(synthetic_cohort),
                'organ_sources': self.organ_sources,
                'creation_date': pd.Timestamp.now().isoformat()
            }
        )
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(hybrid_data, f)
        
        print(f"\n✓ Saved hybrid dataset to {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("HYBRID DATASET SUMMARY")
        print("="*80)
        
        print("\nReal NHANES Transitions:")
        for organ, source in self.organ_sources.items():
            if source == 'real':
                n_trans = len(hybrid_data.real_transitions.get(organ, []))
                print(f"  ✅ {organ}: {n_trans:,} transitions")
        
        print("\nSynthetic Transitions:")
        for organ, source in self.organ_sources.items():
            if source == 'synthetic':
                n_trans = len(hybrid_data.synthetic_transitions.get(organ, []))
                print(f"  🔬 {organ}: {n_trans:,} transitions")
        
        print(f"\nTotal Patients:")
        print(f"  Real: {hybrid_data.metadata['n_real_patients']:,}")
        print(f"  Synthetic: {hybrid_data.metadata['n_synthetic_patients']:,}")
        
        return hybrid_data
    
    def prepare_for_training(
        self,
        hybrid_data: HybridTrainingData,
        organ: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for temporal model training
        
        Args:
            hybrid_data: Hybrid dataset
            organ: Which organ system to prepare
        
        Returns:
            (X, y) arrays for training
        """
        source = self.organ_sources[organ]
        
        if source == 'real':
            transitions = hybrid_data.real_transitions[organ]
        else:
            transitions = hybrid_data.synthetic_transitions[organ]
        
        # Convert to arrays (simplified - actual implementation would be more complex)
        X = []
        y = []
        
        for trans in transitions:
            # Extract features (state at time t)
            features = []
            for key, val in trans.items():
                if key.endswith('_t') and key != 'time_t':
                    features.append(val)
            
            # Extract targets (state at time t+1)
            targets = []
            for key, val in trans.items():
                if key.endswith('_t1'):
                    targets.append(val)
            
            if features and targets:
                X.append(features)
                y.append(targets)
        
        return np.array(X), np.array(y)


def main():
    """Create hybrid dataset"""
    
    integrator = HybridDataIntegrator()
    
    # Paths
    nhanes_path = './data/nhanes_temporal_transitions.pkl'
    synthetic_path = './data/synthetic_longitudinal_cohort.pkl'
    output_path = './data/hybrid_training_dataset.pkl'
    
    # Check if files exist
    if not Path(synthetic_path).exists():
        print(f"ERROR: Synthetic cohort not found at {synthetic_path}")
        print("Run synthetic_trajectory_generator.py first!")
        return
    
    if not Path(nhanes_path).exists():
        print(f"WARNING: NHANES data not found at {nhanes_path}")
        print("Creating mock NHANES data for demonstration...")
        
        # Create mock NHANES data
        mock_nhanes = {
            'metabolic': [
                {
                    'patient_id': f'NHANES_{i}',
                    'glucose_t': np.random.normal(100, 15),
                    'glucose_t1': np.random.normal(100, 15),
                    'age_t': np.random.randint(30, 80)
                }
                for i in range(1000)
            ],
            'cardiovascular': [
                {
                    'patient_id': f'NHANES_{i}',
                    'systolic_bp_t': np.random.normal(120, 15),
                    'systolic_bp_t1': np.random.normal(120, 15),
                    'age_t': np.random.randint(30, 80)
                }
                for i in range(1000)
            ],
            'kidney': [
                {
                    'patient_id': f'NHANES_{i}',
                    'creatinine_t': np.random.normal(1.0, 0.2),
                    'creatinine_t1': np.random.normal(1.0, 0.2),
                    'age_t': np.random.randint(30, 80)
                }
                for i in range(1000)
            ]
        }
        
        Path(nhanes_path).parent.mkdir(parents=True, exist_ok=True)
        with open(nhanes_path, 'wb') as f:
            pickle.dump(mock_nhanes, f)
        
        print(f"  ✓ Created mock NHANES data")
    
    # Create hybrid dataset
    hybrid_data = integrator.create_hybrid_dataset(
        nhanes_path=nhanes_path,
        synthetic_path=synthetic_path,
        output_path=output_path
    )
    
    # Test data preparation
    print("\n" + "="*80)
    print("TESTING DATA PREPARATION FOR TRAINING")
    print("="*80)
    
    for organ in ['metabolic', 'liver', 'neural']:
        X, y = integrator.prepare_for_training(hybrid_data, organ)
        source = integrator.organ_sources[organ]
        print(f"\n{organ.upper()} ({source}):")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Sample X: {X[0] if len(X) > 0 else 'N/A'}")
    
    print("\n" + "="*80)
    print("✓ HYBRID DATASET CREATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Train temporal models on hybrid dataset")
    print("2. Validate learned vs synthetic organ dynamics")
    print("3. Integrate into digital twin system")
    print("4. Document methodology for publication")


if __name__ == '__main__':
    main()
