#!/usr/bin/env python3
"""
Train Temporal Models on Hybrid Dataset

Trains GNN-Transformer models on combination of:
- Real NHANES data (metabolic, cardiovascular, kidney)
- Synthetic physics-informed data (liver, immune, neural, lifestyle)
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid
from organ_simulation.hybrid_data_integrator import HybridTrainingData, SyntheticPatient


class HybridTemporalTrainer:
    """
    Train temporal models on hybrid real+synthetic data
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device = device
        
        # Organ feature dimensions
        self.node_dims = {
            'metabolic': 4,      # glucose, HbA1c, insulin, triglycerides
            'cardiovascular': 5,  # systolic, diastolic, total_chol, HDL, LDL
            'kidney': 2,          # creatinine, BUN
            'liver': 2,           # ALT, AST
            'immune': 1,          # WBC
            'neural': 1,          # cognitive_score
            'lifestyle': 4        # alcohol, exercise, diet, sleep
        }
        
        self.organ_sources = {
            'metabolic': 'real',
            'cardiovascular': 'real',
            'kidney': 'real',
            'liver': 'synthetic',
            'immune': 'synthetic',
            'neural': 'synthetic',
            'lifestyle': 'synthetic'
        }
    
    def load_hybrid_data(self, data_path: str) -> HybridTrainingData:
        """Load hybrid dataset"""
        print(f"Loading hybrid dataset from {data_path}...")
        
        with open(data_path, 'rb') as f:
            hybrid_data = pickle.load(f)
        
        print(f"  ✓ Loaded hybrid dataset")
        print(f"    Real patients: {hybrid_data.metadata['n_real_patients']}")
        print(f"    Synthetic patients: {hybrid_data.metadata['n_synthetic_patients']}")
        
        return hybrid_data
    
    def prepare_training_data(
        self,
        hybrid_data: HybridTrainingData,
        organ: str,
        train_split: float = 0.8
    ) -> Tuple[List, List]:
        """
        Prepare training data for specific organ
        
        Returns:
            (train_data, val_data) lists of transitions
        """
        source = self.organ_sources[organ]
        
        if source == 'real':
            transitions = hybrid_data.real_transitions.get(organ, [])
        else:
            transitions = hybrid_data.synthetic_transitions.get(organ, [])
        
        # Shuffle
        np.random.shuffle(transitions)
        
        # Split
        n_train = int(len(transitions) * train_split)
        train_data = transitions[:n_train]
        val_data = transitions[n_train:]
        
        print(f"\n{organ.upper()} ({source}):")
        print(f"  Total transitions: {len(transitions):,}")
        print(f"  Train: {len(train_data):,}")
        print(f"  Val: {len(val_data):,}")
        
        return train_data, val_data
    
    def create_simple_predictor(self, organ: str) -> nn.Module:
        """
        Create simple MLP predictor for organ dynamics
        
        For now, use simple architecture. Can upgrade to GNN-Transformer later.
        """
        # Fixed input dimension for all organs (padded features)
        input_dim = 10
        output_dim = self.node_dims[organ]
        
        model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
        return model.to(self.device)
    
    def extract_features_targets(
        self,
        transitions: List[Dict],
        organ: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features and targets from transitions"""
        
        X_list = []
        y_list = []
        
        for trans in transitions:
            # Extract state at time t
            features = []
            
            # Organ-specific features
            if organ == 'liver':
                features.extend([trans['ALT_t'], trans['AST_t']])
                features.extend([trans['alcohol_t'], trans['exercise_t'], trans['diet_t']])
                features.append(trans['age_t'])
                # Pad to fixed size
                features.extend([0] * 4)
                
                targets = [trans['ALT_t1'], trans['AST_t1']]
                
            elif organ == 'immune':
                features.append(trans['WBC_t'])
                features.extend([trans['ALT_t'], trans['exercise_t']])
                features.append(trans['age_t'])
                # Pad
                features.extend([0] * 6)
                
                targets = [trans['WBC_t1']]
                
            elif organ == 'neural':
                features.append(trans['cognitive_t'])
                features.extend([trans['exercise_t'], trans['diet_t'], trans['ALT_t']])
                features.append(trans['age_t'])
                # Pad
                features.extend([0] * 5)
                
                targets = [trans['cognitive_t1']]
                
            elif organ == 'lifestyle':
                features.extend([trans['alcohol_t'], trans['exercise_t'], trans['diet_t']])
                features.append(trans['ALT_t'])
                features.append(trans['age_t'])
                # Pad
                features.extend([0] * 5)
                
                targets = [trans['alcohol_t1'], trans['exercise_t1'], trans['diet_t1'], 7.0]
                
            else:
                # For real organs (simplified)
                for key, val in trans.items():
                    if key.endswith('_t') and key != 'time_t':
                        features.append(val)
                
                # Pad to fixed size
                while len(features) < 10:
                    features.append(0)
                features = features[:10]
                
                targets = []
                for key, val in trans.items():
                    if key.endswith('_t1'):
                        targets.append(val)
            
            if features and targets:
                X_list.append(features)
                y_list.append(targets)
        
        X = torch.tensor(X_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)
        
        return X, y
    
    def train_organ_model(
        self,
        organ: str,
        train_data: List[Dict],
        val_data: List[Dict],
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 0.001
    ) -> Dict:
        """
        Train temporal model for specific organ
        
        Returns:
            Training history
        """
        print(f"\nTraining {organ} model...")
        
        # Create model
        model = self.create_simple_predictor(organ)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare data
        X_train, y_train = self.extract_features_targets(train_data, organ)
        X_val, y_val = self.extract_features_targets(val_data, organ)
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        print(f"  Training data: {X_train.shape}")
        print(f"  Validation data: {X_val.shape}")
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            model.train()
            
            # Mini-batch training
            n_batches = len(X_train) // batch_size
            train_losses = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val)
                val_loss = criterion(y_val_pred, y_val).item()
            
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model
        model_path = f'./models/temporal_{organ}_hybrid.pt'
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Saved model to {model_path}")
        
        return history
    
    def train_all_organs(
        self,
        hybrid_data: HybridTrainingData,
        epochs: int = 50
    ) -> Dict:
        """Train models for all organ systems"""
        
        print("="*80)
        print("TRAINING TEMPORAL MODELS ON HYBRID DATASET")
        print("="*80)
        
        all_histories = {}
        
        for organ in ['liver', 'immune', 'neural', 'lifestyle']:
            print(f"\n{'='*80}")
            print(f"ORGAN: {organ.upper()}")
            print(f"Source: {self.organ_sources[organ]}")
            print(f"{'='*80}")
            
            # Prepare data
            train_data, val_data = self.prepare_training_data(hybrid_data, organ)
            
            # Train
            history = self.train_organ_model(
                organ=organ,
                train_data=train_data,
                val_data=val_data,
                epochs=epochs
            )
            
            all_histories[organ] = history
        
        return all_histories
    
    def evaluate_predictions(
        self,
        hybrid_data: HybridTrainingData,
        organ: str,
        n_samples: int = 100
    ):
        """Evaluate model predictions on validation data"""
        
        print(f"\n{'='*80}")
        print(f"EVALUATING {organ.upper()} MODEL")
        print(f"{'='*80}")
        
        # Load model
        model = self.create_simple_predictor(organ)
        model_path = f'./models/temporal_{organ}_hybrid.pt'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Get validation data
        _, val_data = self.prepare_training_data(hybrid_data, organ, train_split=0.8)
        val_data = val_data[:n_samples]
        
        X_val, y_val = self.extract_features_targets(val_data, organ)
        X_val = X_val.to(self.device)
        
        # Predict
        with torch.no_grad():
            y_pred = model(X_val).cpu().numpy()
        
        y_val = y_val.numpy()
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred - y_val), axis=0)
        rmse = np.sqrt(np.mean((y_pred - y_val)**2, axis=0))
        
        print(f"\nPrediction Metrics (n={n_samples}):")
        
        if organ == 'liver':
            print(f"  ALT - MAE: {mae[0]:.2f} U/L, RMSE: {rmse[0]:.2f} U/L")
            print(f"  AST - MAE: {mae[1]:.2f} U/L, RMSE: {rmse[1]:.2f} U/L")
        elif organ == 'immune':
            print(f"  WBC - MAE: {mae[0]:.2f} K/μL, RMSE: {rmse[0]:.2f} K/μL")
        elif organ == 'neural':
            print(f"  Cognitive - MAE: {mae[0]:.3f}, RMSE: {rmse[0]:.3f}")
        elif organ == 'lifestyle':
            print(f"  Alcohol - MAE: {mae[0]:.3f}, RMSE: {rmse[0]:.3f}")
            print(f"  Exercise - MAE: {mae[1]:.3f}, RMSE: {rmse[1]:.3f}")
            print(f"  Diet - MAE: {mae[2]:.3f}, RMSE: {rmse[2]:.3f}")
        
        # Show examples
        print(f"\nExample Predictions:")
        print(f"{'Actual':<20} {'Predicted':<20} {'Error':<15}")
        print("-" * 55)
        
        for i in range(min(5, len(y_val))):
            actual = y_val[i]
            predicted = y_pred[i]
            error = np.abs(actual - predicted)
            
            print(f"{str(actual):<20} {str(predicted):<20} {str(error):<15}")


def main():
    """Train hybrid models"""
    
    trainer = HybridTemporalTrainer(
        hidden_dim=128,
        n_heads=4,
        n_layers=3
    )
    
    # Load hybrid dataset
    hybrid_data = trainer.load_hybrid_data('./data/hybrid_training_dataset.pkl')
    
    # Train all organs
    histories = trainer.train_all_organs(hybrid_data, epochs=50)
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for organ in ['liver', 'immune', 'neural', 'lifestyle']:
        trainer.evaluate_predictions(hybrid_data, organ, n_samples=100)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print("\nTrained models saved to ./models/")
    print("  - temporal_liver_hybrid.pt")
    print("  - temporal_immune_hybrid.pt")
    print("  - temporal_neural_hybrid.pt")
    print("  - temporal_lifestyle_hybrid.pt")
    
    print("\nNext steps:")
    print("1. Integrate trained models into digital twin system")
    print("2. Test full system with example patients")
    print("3. Create validation report")
    print("4. Prepare methodology paper")


if __name__ == '__main__':
    main()
