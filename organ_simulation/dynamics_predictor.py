#!/usr/bin/env python3
"""
Data-Driven Organ Dynamics Predictor

Learns how organs change over time from real patient data.
NO hand-coded parameters - everything learned from NHANES transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class OrganDynamicsPredictor(nn.Module):
    """
    Learns organ state evolution from patient temporal data
    
    Training: Given (organs[t], lifestyle[t]) → predict organs[t+1]
    Uses pretrained GNN + Transformer embeddings
    """
    
    def __init__(
        self,
        gnn_hidden_dim: int = 64,
        transformer_dim: int = 512,
        num_organs: int = 7,
        lifestyle_dim: int = 5,
        organ_feature_dims: Dict[str, int] = None
    ):
        super().__init__()
        
        self.gnn_hidden_dim = gnn_hidden_dim
        self.transformer_dim = transformer_dim
        self.num_organs = num_organs
        
        if organ_feature_dims is None:
            organ_feature_dims = {
                'metabolic': 4,
                'cardiovascular': 5,
                'liver': 2,
                'kidney': 2,
                'immune': 1,
                'neural': 1,
                'lifestyle': 4
            }
        self.organ_feature_dims = organ_feature_dims
        
        # Fusion layer: Combine GNN organ interactions + Transformer temporal context
        fusion_input_dim = (gnn_hidden_dim * num_organs) + transformer_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Organ-specific delta predictors
        # Predicts CHANGE in each organ (not absolute values)
        self.delta_predictors = nn.ModuleDict()
        for organ, dim in organ_feature_dims.items():
            self.delta_predictors[organ] = nn.Sequential(
                nn.Linear(256 + lifestyle_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, dim)  # Predict delta for each feature
            )
        
        # Uncertainty estimator (for confidence intervals)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, sum(organ_feature_dims.values()))
        )
    
    def forward(
        self,
        gnn_embeddings: torch.Tensor,  # [batch, num_organs, gnn_hidden_dim]
        temporal_context: torch.Tensor,  # [batch, transformer_dim]
        lifestyle: torch.Tensor  # [batch, lifestyle_dim]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict organ state changes
        
        Args:
            gnn_embeddings: GNN output (organ interactions)
            temporal_context: Transformer output (temporal patterns)
            lifestyle: Current lifestyle factors
        
        Returns:
            deltas: Predicted change for each organ
            uncertainty: Prediction uncertainty (std dev)
        """
        batch_size = gnn_embeddings.shape[0]
        
        # Flatten GNN embeddings: [batch, num_organs * gnn_hidden_dim]
        gnn_flat = gnn_embeddings.reshape(batch_size, -1)
        
        # Fuse multi-modal context
        combined = torch.cat([gnn_flat, temporal_context], dim=-1)
        shared_features = self.fusion(combined)
        
        # Predict organ-specific deltas
        deltas = {}
        for organ, predictor in self.delta_predictors.items():
            organ_input = torch.cat([shared_features, lifestyle], dim=-1)
            deltas[organ] = predictor(organ_input)
        
        # Estimate uncertainty
        uncertainty = F.softplus(self.uncertainty_head(shared_features))
        
        return deltas, uncertainty
    
    def predict_trajectory(
        self,
        initial_organs: Dict[str, torch.Tensor],
        lifestyle_sequence: torch.Tensor,
        gnn_model,
        transformer_model,
        n_steps: int,
        return_uncertainty: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Simulate forward trajectory
        
        Args:
            initial_organs: Starting organ states
            lifestyle_sequence: Lifestyle at each timestep [n_steps, lifestyle_dim]
            gnn_model: Pretrained GNN
            transformer_model: Pretrained Transformer
            n_steps: Number of months to simulate
            return_uncertainty: Whether to return confidence intervals
        
        Returns:
            trajectory: List of organ states at each timestep
        """
        trajectory = [initial_organs]
        uncertainties = [] if return_uncertainty else None
        
        current_organs = {k: v.clone() for k, v in initial_organs.items()}
        
        for step in range(n_steps):
            # Get GNN embeddings (organ interactions)
            gnn_emb = gnn_model(
                current_organs,
                gnn_model.edge_index,
                return_hidden=True
            )
            # Stack organ embeddings: [1, num_organs, hidden_dim]
            gnn_emb_stacked = torch.stack([
                gnn_emb[organ] for organ in sorted(self.organ_feature_dims.keys())
            ], dim=1)
            
            # Get Transformer context (temporal patterns)
            # Use last 12 months of history
            history_window = trajectory[-12:] if len(trajectory) >= 12 else trajectory
            if len(history_window) > 1:
                # Convert history to tensor format
                history_tensor = self._history_to_tensor(history_window)
                temporal_emb, _ = transformer_model(
                    history_tensor.unsqueeze(0),  # Add batch dim
                    time_deltas=torch.ones(1, len(history_window))
                )
            else:
                # Initial state - no history
                temporal_emb = torch.zeros(1, self.transformer_dim)
            
            # Get lifestyle for this timestep
            lifestyle = lifestyle_sequence[step:step+1]
            
            # Predict deltas
            deltas, uncertainty = self.forward(
                gnn_emb_stacked,
                temporal_emb,
                lifestyle
            )
            
            # Update organ states
            next_organs = {}
            for organ in current_organs.keys():
                next_organs[organ] = current_organs[organ] + deltas[organ]
            
            trajectory.append(next_organs)
            if return_uncertainty:
                uncertainties.append(uncertainty)
            
            current_organs = next_organs
        
        if return_uncertainty:
            return trajectory, uncertainties
        return trajectory
    
    def _history_to_tensor(self, history: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Convert history of organ states to tensor format for Transformer"""
        # Stack all organs for each timestep
        timesteps = []
        for state in history:
            organs_flat = torch.cat([
                state[organ] for organ in sorted(self.organ_feature_dims.keys())
            ], dim=-1)
            timesteps.append(organs_flat)
        return torch.stack(timesteps, dim=0)


class DynamicsTrainer:
    """Train dynamics predictor on NHANES temporal transitions"""
    
    def __init__(
        self,
        dynamics_model: OrganDynamicsPredictor,
        gnn_model,
        transformer_model,
        device: str = 'cuda'
    ):
        self.dynamics_model = dynamics_model.to(device)
        self.gnn_model = gnn_model.to(device)
        self.transformer_model = transformer_model.to(device)
        self.device = device
        
        # Freeze pretrained models
        for param in self.gnn_model.parameters():
            param.requires_grad = False
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.dynamics_model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
    
    def train_on_transitions(
        self,
        patient_data: List[Dict],
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Train on real patient transitions
        
        patient_data format:
        [
            {
                'patient_id': 12345,
                'trajectory': [
                    {'organs': {...}, 'lifestyle': {...}, 'time': 0},
                    {'organs': {...}, 'lifestyle': {...}, 'time': 12},
                    ...
                ]
            },
            ...
        ]
        """
        print(f"Training dynamics predictor on {len(patient_data)} patients...")
        
        # Extract all transitions
        transitions = self._extract_transitions(patient_data)
        print(f"Found {len(transitions)} temporal transitions")
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle transitions
            np.random.shuffle(transitions)
            
            for i in range(0, len(transitions), batch_size):
                batch = transitions[i:i+batch_size]
                loss = self._train_batch(batch)
                total_loss += loss
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Transition prediction loss = {avg_loss:.4f}")
    
    def _extract_transitions(self, patient_data: List[Dict]) -> List[Dict]:
        """Extract all state transitions from patient trajectories"""
        transitions = []
        
        for patient in patient_data:
            trajectory = patient['trajectory']
            
            # For each consecutive pair of timepoints
            for t in range(len(trajectory) - 1):
                current = trajectory[t]
                next_state = trajectory[t + 1]
                
                transitions.append({
                    'current_organs': current['organs'],
                    'lifestyle': current['lifestyle'],
                    'next_organs': next_state['organs'],
                    'history': trajectory[:t+1]  # For temporal context
                })
        
        return transitions
    
    def _train_batch(self, batch: List[Dict]) -> float:
        """Train on a batch of transitions"""
        self.optimizer.zero_grad()
        
        batch_loss = 0
        
        for transition in batch:
            current_organs = {
                k: v.to(self.device) for k, v in transition['current_organs'].items()
            }
            lifestyle = transition['lifestyle'].to(self.device)
            actual_next = {
                k: v.to(self.device) for k, v in transition['next_organs'].items()
            }
            
            # Get GNN embeddings
            with torch.no_grad():
                gnn_emb = self.gnn_model(
                    current_organs,
                    self.gnn_model.edge_index,
                    return_hidden=True
                )
                gnn_emb_stacked = torch.stack([
                    gnn_emb[organ] 
                    for organ in sorted(self.dynamics_model.organ_feature_dims.keys())
                ], dim=1).unsqueeze(0)
                
                # Get temporal context
                # TODO: Implement history encoding
                temporal_emb = torch.zeros(1, self.dynamics_model.transformer_dim).to(self.device)
            
            # Predict deltas
            predicted_deltas, uncertainty = self.dynamics_model(
                gnn_emb_stacked,
                temporal_emb,
                lifestyle.unsqueeze(0)
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
        torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 1.0)
        self.optimizer.step()
        
        return batch_loss.item()
