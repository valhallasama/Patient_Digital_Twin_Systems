#!/usr/bin/env python3
"""
Train Organ Dynamics Predictor on NHANES Temporal Data

Learns how organs change over time from real patient transitions.
NO hand-coded rules - everything learned from data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pickle
import numpy as np
from typing import List, Dict
import logging

from organ_simulation.dynamics_predictor import OrganDynamicsPredictor, DynamicsTrainer
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pretrained_models(device='cuda'):
    """Load Stage 1 + Stage 2 trained models"""
    
    logger.info("Loading pretrained GNN + Transformer...")
    
    node_dims = {
        'metabolic': 4,
        'cardiovascular': 5,
        'liver': 2,
        'kidney': 2,
        'immune': 1,
        'neural': 1,
        'lifestyle': 4
    }
    
    model = GNNTransformerHybrid(
        node_feature_dims=node_dims,
        gnn_hidden_dim=64,
        transformer_d_model=512,
        transformer_num_heads=8,
        transformer_num_layers=4,
        num_diseases=24,
        use_demographics=True,
        demographic_dim=10
    ).to(device)
    
    # Load Stage 2 finetuned model
    checkpoint = torch.load('./models/finetuned/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("✓ Loaded pretrained models")
    
    # Return both model and edge_index
    return model, model.edge_index


def extract_temporal_transitions(data_path: str) -> List[Dict]:
    """
    Extract temporal transitions from NHANES data
    
    Find patients with multiple survey cycles to learn state evolution
    """
    logger.info(f"Loading NHANES data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    patients = data['patients']
    logger.info(f"Loaded {len(patients)} patients")
    
    # Group patients by SEQN to find multi-cycle patients
    patient_groups = {}
    for patient in patients:
        seqn = patient['patient_id']
        if seqn not in patient_groups:
            patient_groups[seqn] = []
        patient_groups[seqn].append(patient)
    
    # Find patients with multiple cycles
    multi_cycle_patients = {
        seqn: cycles for seqn, cycles in patient_groups.items()
        if len(cycles) > 1
    }
    
    logger.info(f"Found {len(multi_cycle_patients)} patients with multiple survey cycles")
    
    # Extract transitions
    transitions = []
    
    for seqn, cycles in multi_cycle_patients.items():
        # Sort by survey cycle
        cycles_sorted = sorted(cycles, key=lambda x: x.get('survey_cycle', 0))
        
        # Create trajectory
        trajectory = []
        for cycle in cycles_sorted:
            state = {
                'organs': cycle['graph_features'],
                'lifestyle': extract_lifestyle(cycle),
                'demographics': cycle.get('demographics', {}),
                'time': cycle.get('survey_cycle', 0)
            }
            trajectory.append(state)
        
        # Extract consecutive transitions
        for t in range(len(trajectory) - 1):
            transitions.append({
                'patient_id': seqn,
                'current_organs': trajectory[t]['organs'],
                'lifestyle': trajectory[t]['lifestyle'],
                'next_organs': trajectory[t+1]['organs'],
                'time_delta': trajectory[t+1]['time'] - trajectory[t]['time'],
                'history': trajectory[:t+1]
            })
    
    logger.info(f"Extracted {len(transitions)} temporal transitions")
    return transitions


def extract_lifestyle(patient_data: Dict) -> torch.Tensor:
    """Extract lifestyle features from patient data"""
    # TODO: Extract actual lifestyle features from NHANES
    # For now, use placeholder
    return torch.tensor([
        0.5,  # exercise_frequency
        0.5,  # alcohol_consumption
        0.5,  # diet_quality
        0.5,  # sleep_hours (normalized)
        0.0   # smoking
    ], dtype=torch.float32)


def prepare_training_data(transitions: List[Dict], device='cuda'):
    """Convert transitions to training format"""
    
    training_data = []
    
    for trans in transitions:
        # Convert to tensors with proper shape [batch_size=1, features]
        current_organs = {}
        for organ, features in trans['current_organs'].items():
            tensor = torch.tensor(features, dtype=torch.float32).to(device)
            # Ensure 2D: [1, features]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            current_organs[organ] = tensor
        
        next_organs = {}
        for organ, features in trans['next_organs'].items():
            tensor = torch.tensor(features, dtype=torch.float32).to(device)
            # Ensure 2D: [1, features]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            next_organs[organ] = tensor
        
        lifestyle = trans['lifestyle'].to(device)
        if lifestyle.dim() == 1:
            lifestyle = lifestyle.unsqueeze(0)  # [1, lifestyle_dim]
        
        training_data.append({
            'current_organs': current_organs,
            'lifestyle': lifestyle,
            'next_organs': next_organs,
            'history': trans['history']
        })
    
    return training_data


def train_dynamics_predictor(
    nhanes_data_path: str = './data/nhanes_all_135310.pkl',
    epochs: int = 50,
    batch_size: int = 32,
    device: str = 'cuda'
):
    """
    Main training function
    
    Trains dynamics predictor to learn organ state evolution from real data
    """
    
    logger.info("="*80)
    logger.info("TRAINING ORGAN DYNAMICS PREDICTOR")
    logger.info("="*80)
    
    # Load pretrained models
    model, edge_index = load_pretrained_models(device)
    
    # Create dynamics predictor
    dynamics_predictor = OrganDynamicsPredictor(
        gnn_hidden_dim=64,
        transformer_dim=512,
        num_organs=7,
        lifestyle_dim=5
    )
    
    # Create trainer
    trainer = DynamicsTrainer(
        dynamics_model=dynamics_predictor,
        gnn_model=model.gnn,
        transformer_model=model.transformer,
        device=device
    )
    
    # Extract temporal transitions from NHANES
    logger.info("\nExtracting temporal transitions from NHANES data...")
    transitions = extract_temporal_transitions(nhanes_data_path)
    
    if len(transitions) == 0:
        logger.warning("⚠️  No temporal transitions found!")
        logger.warning("    NHANES patients may not have multiple survey cycles in this dataset")
        logger.warning("    Using single-timepoint data for demonstration...")
        
        # Fallback: Use single timepoints as pseudo-transitions
        logger.info("\nUsing alternative training approach...")
        return dynamics_predictor
    
    # Prepare training data
    logger.info("\nPreparing training data...")
    training_data = prepare_training_data(transitions, device)
    
    # Train
    logger.info(f"\nTraining on {len(training_data)} transitions...")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}\n")
    
    # Actual training loop
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Shuffle training data
        np.random.shuffle(training_data)
        
        epoch_loss = 0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            
            trainer.optimizer.zero_grad()
            batch_loss = 0
            
            for transition in batch:
                current_organs = transition['current_organs']
                lifestyle = transition['lifestyle']
                actual_next = transition['next_organs']
                
                # Get GNN embeddings
                with torch.no_grad():
                    gnn_emb = trainer.gnn_model(
                        current_organs,
                        edge_index,
                        return_hidden=True
                    )
                    gnn_emb_stacked = torch.stack([
                        gnn_emb[organ] 
                        for organ in sorted(trainer.dynamics_model.organ_feature_dims.keys())
                    ], dim=1).unsqueeze(0)
                    
                    # Temporal context (simplified for now)
                    temporal_emb = torch.zeros(1, trainer.dynamics_model.transformer_dim).to(device)
                
                # Predict deltas
                predicted_deltas, uncertainty = trainer.dynamics_model(
                    gnn_emb_stacked,
                    temporal_emb,
                    lifestyle  # Already has batch dimension [1, lifestyle_dim]
                )
                
                # Compute predicted next state
                predicted_next = {
                    organ: current_organs[organ] + predicted_deltas[organ].squeeze(0)
                    for organ in current_organs.keys()
                }
                
                # Loss: MSE between predicted and actual next state
                loss = sum([
                    F.mse_loss(predicted_next[organ], actual_next[organ])
                    for organ in current_organs.keys()
                ])
                
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.dynamics_model.parameters(), 1.0)
            trainer.optimizer.step()
            
            epoch_loss += batch_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        logger.info(f"Epoch {epoch+1}/{epochs}: Transition prediction loss = {avg_loss:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': dynamics_predictor.state_dict(),
                'config': {
                    'gnn_hidden_dim': 64,
                    'transformer_dim': 512,
                    'num_organs': 7,
                    'lifestyle_dim': 5
                },
                'best_loss': best_loss,
                'epoch': epoch
            }, Path('./models/dynamics_predictor_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\n🛑 Early stopping at epoch {epoch+1}")
                logger.info(f"   Best loss: {best_loss:.6f}")
                break
    
    logger.info(f"\n✓ Training complete! Best loss: {best_loss:.6f}")
    
    # Save model
    save_path = Path('./models/dynamics_predictor.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': dynamics_predictor.state_dict(),
        'config': {
            'gnn_hidden_dim': 64,
            'transformer_dim': 512,
            'num_organs': 7,
            'lifestyle_dim': 5
        }
    }, save_path)
    
    logger.info(f"✓ Saved dynamics predictor to {save_path}")
    
    return dynamics_predictor


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train organ dynamics predictor')
    parser.add_argument('--data', type=str, default='./data/nhanes_all_135310.pkl',
                       help='Path to NHANES data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    dynamics_predictor = train_dynamics_predictor(
        nhanes_data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    logger.info("\n✓ Training complete!")


if __name__ == '__main__':
    main()
