#!/usr/bin/env python3
"""
Stateful Organ Agents for True Digital Twin Simulation

Converts static prediction into dynamic simulation with:
- Agent-based organ states with memory
- Stochastic state evolution
- Feedback loops for temporal propagation
- Emergent multi-organ dynamics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OrganState:
    """
    State representation for a single organ at a timestep
    
    Includes:
    - Current features (biomarkers, function)
    - Hidden state (memory from previous timesteps)
    - Uncertainty (for stochastic simulation)
    """
    features: torch.Tensor  # [feature_dim]
    hidden: torch.Tensor    # [hidden_dim]
    uncertainty: torch.Tensor  # [feature_dim]
    timestep: int
    
    def to_dict(self) -> Dict:
        return {
            'features': self.features.cpu().numpy(),
            'hidden': self.hidden.cpu().numpy(),
            'uncertainty': self.uncertainty.cpu().numpy(),
            'timestep': self.timestep
        }


class OrganAgent(nn.Module):
    """
    Agent representing a single organ system
    
    Maintains internal state and evolves over time based on:
    - Internal dynamics (organ-specific rules)
    - External inputs (from other organs via GNN)
    - Stochastic perturbations (biological variability)
    """
    
    def __init__(
        self,
        organ_name: str,
        feature_dim: int,
        hidden_dim: int = 64,
        use_stochastic: bool = True
    ):
        super().__init__()
        
        self.organ_name = organ_name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_stochastic = use_stochastic
        
        # State transition network (LSTM-like)
        self.state_transition = nn.LSTMCell(
            input_size=feature_dim,
            hidden_size=hidden_dim
        )
        
        # Feature prediction network
        self.feature_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Organ-specific dynamics (learned residuals)
        self.dynamics_network = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def initialize_state(
        self,
        initial_features: torch.Tensor,
        batch_size: int = 1
    ) -> OrganState:
        """Initialize organ state from baseline features"""
        device = initial_features.device
        
        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        cell = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Run one step to get initial hidden state
        hidden, cell = self.state_transition(initial_features, (hidden, cell))
        
        # Estimate initial uncertainty
        uncertainty = self.uncertainty_estimator(hidden)
        
        return OrganState(
            features=initial_features,
            hidden=hidden,
            uncertainty=uncertainty,
            timestep=0
        )
    
    def step(
        self,
        current_state: OrganState,
        external_input: torch.Tensor,
        time_delta: float = 1.0,
        stochastic: bool = True
    ) -> OrganState:
        """
        Evolve organ state forward by one timestep
        
        Args:
            current_state: Current organ state
            external_input: Input from other organs (via GNN)
            time_delta: Time elapsed (in months)
            stochastic: Whether to add stochastic perturbations
            
        Returns:
            New organ state
        """
        # Combine current features with external input
        combined_input = current_state.features + external_input
        
        # Update hidden state (memory)
        hidden, cell = self.state_transition(
            combined_input,
            (current_state.hidden, current_state.hidden)  # Use hidden as cell state
        )
        
        # Predict feature changes
        feature_delta = self.feature_predictor(hidden)
        
        # Add organ-specific dynamics
        dynamics_input = torch.cat([current_state.features, hidden], dim=-1)
        dynamics_delta = self.dynamics_network(dynamics_input)
        
        # Combine predictions (weighted by time_delta)
        total_delta = (feature_delta + dynamics_delta) * time_delta
        
        # Add stochastic perturbations
        if stochastic and self.use_stochastic:
            uncertainty = self.uncertainty_estimator(hidden)
            noise = torch.randn_like(total_delta) * uncertainty * np.sqrt(time_delta)
            total_delta = total_delta + noise
        else:
            uncertainty = self.uncertainty_estimator(hidden)
        
        # Update features
        new_features = current_state.features + total_delta
        
        return OrganState(
            features=new_features,
            hidden=hidden,
            uncertainty=uncertainty,
            timestep=current_state.timestep + 1
        )


class MultiOrganSimulator(nn.Module):
    """
    Multi-organ digital twin simulator with agent-based dynamics
    
    Combines:
    - Individual organ agents (with memory)
    - GNN for organ-organ interactions
    - Temporal Transformer for long-range dependencies
    - Feedback loops for simulation
    """
    
    def __init__(
        self,
        organ_configs: Dict[str, int],  # organ_name -> feature_dim
        gnn_model: nn.Module,
        transformer_model: nn.Module,
        hidden_dim: int = 64,
        use_stochastic: bool = True
    ):
        super().__init__()
        
        self.organ_names = sorted(organ_configs.keys())
        self.gnn = gnn_model
        self.transformer = transformer_model
        
        # Create organ agents
        self.agents = nn.ModuleDict({
            organ: OrganAgent(
                organ_name=organ,
                feature_dim=dim,
                hidden_dim=hidden_dim,
                use_stochastic=use_stochastic
            )
            for organ, dim in organ_configs.items()
        })
    
    def initialize_simulation(
        self,
        initial_features: Dict[str, torch.Tensor],
        batch_size: int = 1
    ) -> Dict[str, OrganState]:
        """Initialize all organ states"""
        states = {}
        for organ in self.organ_names:
            states[organ] = self.agents[organ].initialize_state(
                initial_features[organ],
                batch_size
            )
        return states
    
    def simulate_step(
        self,
        current_states: Dict[str, OrganState],
        edge_index: torch.Tensor,
        time_delta: float = 1.0,
        stochastic: bool = True
    ) -> Dict[str, OrganState]:
        """
        Simulate one timestep with feedback
        
        Process:
        1. Extract current features from all organs
        2. Run GNN to get organ-organ interactions
        3. Each agent updates its state based on GNN output
        4. Return new states
        """
        # Extract current features
        current_features = {
            organ: state.features
            for organ, state in current_states.items()
        }
        
        # Run GNN to get organ interactions
        gnn_outputs = self.gnn(current_features, edge_index)
        
        # Update each organ state
        new_states = {}
        for organ in self.organ_names:
            external_input = gnn_outputs[organ] - current_features[organ]
            
            new_states[organ] = self.agents[organ].step(
                current_state=current_states[organ],
                external_input=external_input,
                time_delta=time_delta,
                stochastic=stochastic
            )
        
        return new_states
    
    def simulate_trajectory(
        self,
        initial_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        num_steps: int = 60,
        time_deltas: Optional[List[float]] = None,
        stochastic: bool = True
    ) -> Tuple[List[Dict[str, OrganState]], torch.Tensor]:
        """
        Simulate full trajectory with feedback loops
        
        Args:
            initial_features: Initial organ features
            edge_index: Organ graph connectivity
            num_steps: Number of simulation steps
            time_deltas: Time between steps (months), default 1.0
            stochastic: Use stochastic dynamics
            
        Returns:
            trajectory: List of organ states at each timestep
            attention_weights: Temporal attention from Transformer
        """
        if time_deltas is None:
            time_deltas = [1.0] * num_steps
        
        batch_size = initial_features[self.organ_names[0]].shape[0]
        
        # Initialize
        states = self.initialize_simulation(initial_features, batch_size)
        trajectory = [states]
        
        # Simulate forward
        for step in range(num_steps):
            new_states = self.simulate_step(
                current_states=states,
                edge_index=edge_index,
                time_delta=time_deltas[step],
                stochastic=stochastic
            )
            trajectory.append(new_states)
            states = new_states
        
        # Extract feature sequences for Transformer
        organ_sequences = {}
        for organ in self.organ_names:
            organ_sequences[organ] = torch.stack([
                traj[organ].features for traj in trajectory
            ], dim=1)  # [batch, seq_len, feature_dim]
        
        # Run Transformer to get attention weights
        time_delta_tensor = torch.tensor(
            [0.0] + time_deltas,
            device=initial_features[self.organ_names[0]].device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Stack organ embeddings
        organ_embeddings = torch.stack([
            organ_sequences[organ] for organ in self.organ_names
        ], dim=2)  # [batch, seq_len, num_organs, feature_dim]
        
        # Get Transformer attention
        _, attention_info = self.transformer(
            organ_embeddings.reshape(batch_size, num_steps + 1, -1),
            time_delta_tensor
        )
        
        return trajectory, attention_info
    
    def predict_disease_risk(
        self,
        trajectory: List[Dict[str, OrganState]],
        disease_predictor: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Predict disease risks from simulated trajectory
        
        Args:
            trajectory: Simulated organ states over time
            disease_predictor: Multi-disease prediction head
            
        Returns:
            Disease predictions at each timestep
        """
        predictions = []
        
        for states in trajectory:
            # Extract features
            features = torch.cat([
                states[organ].features for organ in self.organ_names
            ], dim=-1)
            
            # Predict diseases
            pred = disease_predictor(features)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = {
            key: torch.stack([p[key] for p in predictions], dim=1)
            for key in predictions[0].keys()
        }
        
        return stacked_preds


# Example usage
if __name__ == '__main__':
    from graph_learning.organ_gnn import OrganGraphNetwork
    from graph_learning.temporal_transformer import TemporalTransformerEncoder
    
    # Organ configurations
    organ_configs = {
        'metabolic': 4,
        'cardiovascular': 5,
        'liver': 2,
        'kidney': 2,
        'immune': 1,
        'neural': 1,
        'lifestyle': 4
    }
    
    # Create GNN
    gnn = OrganGraphNetwork(
        node_feature_dims=organ_configs,
        hidden_dim=64,
        num_gat_layers=2
    )
    
    # Create Transformer
    transformer = TemporalTransformerEncoder(
        d_model=128,
        num_heads=8,
        num_layers=4,
        num_organs=7
    )
    
    # Create simulator
    simulator = MultiOrganSimulator(
        organ_configs=organ_configs,
        gnn_model=gnn,
        transformer_model=transformer,
        hidden_dim=64,
        use_stochastic=True
    )
    
    # Initialize with dummy data
    batch_size = 2
    initial_features = {
        organ: torch.randn(batch_size, dim)
        for organ, dim in organ_configs.items()
    }
    
    # Create edge index (fully connected)
    num_organs = len(organ_configs)
    edge_index = torch.tensor([
        [i, j] for i in range(num_organs) for j in range(num_organs) if i != j
    ]).t()
    
    # Simulate trajectory
    print("Simulating 60-month trajectory...")
    trajectory, attention = simulator.simulate_trajectory(
        initial_features=initial_features,
        edge_index=edge_index,
        num_steps=60,
        stochastic=True
    )
    
    print(f"✓ Simulated {len(trajectory)} timesteps")
    print(f"  Final metabolic features: {trajectory[-1]['metabolic'].features[0, :3]}")
    print(f"  Final uncertainty: {trajectory[-1]['metabolic'].uncertainty[0, :3]}")
    
    print("\n✓ Stateful organ agents implemented!")
    print("  - Agent-based state evolution")
    print("  - Memory via LSTM cells")
    print("  - Stochastic dynamics")
    print("  - Feedback loops via GNN")
