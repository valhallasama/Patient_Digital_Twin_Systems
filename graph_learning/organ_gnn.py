#!/usr/bin/env python3
"""
Organ Graph Neural Network
Graph-based learning layer for multi-organ interactions

Architecture:
- Nodes: 7 organ systems (metabolic, cardiovascular, liver, kidney, immune, neural, lifestyle)
- Edges: Learned interaction strengths (constrained by physiology)
- Message passing: GAT (Graph Attention Network) for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OrganGraphNetwork(nn.Module):
    """
    Graph Neural Network for multi-organ digital twin
    
    Combines:
    - Mechanistic physiology (hard constraints)
    - Learned interactions (data-driven refinement)
    """
    
    # Define organ system indices
    ORGAN_INDICES = {
        'metabolic': 0,
        'cardiovascular': 1,
        'liver': 2,
        'kidney': 3,
        'immune': 4,
        'neural': 5,
        'lifestyle': 6
    }
    
    # Known physiological edges (from medical knowledge)
    KNOWN_EDGES = [
        ('metabolic', 'cardiovascular'),  # Insulin resistance → BP
        ('metabolic', 'liver'),  # Insulin resistance → liver fat
        ('liver', 'cardiovascular'),  # Liver fat → LDL
        ('liver', 'immune'),  # Liver inflammation → systemic inflammation
        ('immune', 'cardiovascular'),  # Inflammation → vascular damage
        ('cardiovascular', 'kidney'),  # Hypertension → kidney damage
        ('metabolic', 'kidney'),  # Diabetes → kidney damage
        ('lifestyle', 'metabolic'),  # Exercise → insulin sensitivity
        ('lifestyle', 'cardiovascular'),  # Exercise → BP
        ('lifestyle', 'immune'),  # Sleep → inflammation
    ]
    
    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 64,
        num_attention_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_physics_constraints: bool = True
    ):
        """
        Initialize organ graph network
        
        Args:
            node_feature_dims: Dictionary mapping organ name to feature dimension
            hidden_dim: Hidden layer dimension
            num_attention_heads: Number of attention heads in GAT
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_physics_constraints: Whether to enforce physiological constraints
        """
        super(OrganGraphNetwork, self).__init__()
        
        self.node_feature_dims = node_feature_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_attention_heads
        self.num_layers = num_layers
        self.use_physics_constraints = use_physics_constraints
        
        # Input projection layers (per organ type)
        self.input_projections = nn.ModuleDict({
            organ: nn.Linear(dim, hidden_dim)
            for organ, dim in node_feature_dims.items()
        })
        
        # GAT layers for message passing
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim // num_attention_heads,
                heads=num_attention_heads,
                dropout=dropout,
                add_self_loops=True,
                concat=True
            )
            for i in range(num_layers)
        ])
        
        # Output projection layers (per organ type)
        self.output_projections = nn.ModuleDict({
            organ: nn.Linear(hidden_dim, dim)
            for organ, dim in node_feature_dims.items()
        })
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization (all layers output hidden_dim due to concat=True)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Physics-informed edge weights (learnable but constrained)
        if use_physics_constraints:
            self._initialize_physics_constraints()
    
    def _initialize_physics_constraints(self):
        """Initialize edge weights based on known physiology"""
        # Create adjacency matrix for known edges
        num_organs = len(self.ORGAN_INDICES)
        self.edge_mask = torch.zeros(num_organs, num_organs)
        
        for source, target in self.KNOWN_EDGES:
            src_idx = self.ORGAN_INDICES[source]
            tgt_idx = self.ORGAN_INDICES[target]
            self.edge_mask[src_idx, tgt_idx] = 1.0
            # Also allow reverse direction (bidirectional)
            self.edge_mask[tgt_idx, src_idx] = 1.0
        
        # Allow self-loops
        for i in range(num_organs):
            self.edge_mask[i, i] = 1.0
    
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_hidden: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through organ graph
        
        Args:
            node_features: Dictionary mapping organ name to feature tensor
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for each node (for batched graphs)
            return_hidden: If True, return hidden_dim embeddings; if False, project back to original dims
        
        Returns:
            Updated node features per organ
        """
        # Project input features to hidden dimension
        x_list = []
        organ_types = []
        
        for organ in sorted(self.ORGAN_INDICES.keys(), key=lambda k: self.ORGAN_INDICES[k]):
            if organ in node_features:
                x = self.input_projections[organ](node_features[organ])
                x_list.append(x)
                organ_types.append(organ)
        
        # Stack all node features
        x = torch.cat(x_list, dim=0)  # [num_nodes, hidden_dim]
        
        # Apply physics constraints to edges if enabled
        if self.use_physics_constraints:
            edge_index = self._apply_physics_constraints(edge_index)
        
        # Message passing through GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # GAT layer
            x_new = gat(x, edge_index)
            
            # Residual connection (only if dimensions match)
            if i > 0 and x_new.shape == x.shape:
                x_new = x_new + x
            
            # Layer normalization
            x_new = norm(x_new)
            
            # Activation
            x_new = F.elu(x_new)
            
            # Dropout
            x_new = self.dropout(x_new)
            
            x = x_new
        
        # Split back into organs
        outputs = {}
        start_idx = 0
        
        for organ in organ_types:
            num_nodes = node_features[organ].shape[0]
            end_idx = start_idx + num_nodes
            
            organ_features = x[start_idx:end_idx]
            
            # Return hidden embeddings (for transformer) or project back (for reconstruction)
            if return_hidden:
                outputs[organ] = organ_features  # [batch, hidden_dim]
            else:
                outputs[organ] = self.output_projections[organ](organ_features)
            
            start_idx = end_idx
        
        return outputs
    
    def _apply_physics_constraints(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Filter edges to only allow physiologically plausible connections
        
        Args:
            edge_index: [2, num_edges]
        
        Returns:
            Filtered edge_index
        """
        # Convert edge_index to mask
        device = edge_index.device
        edge_mask = self.edge_mask.to(device)
        
        # Filter edges
        valid_edges = []
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            if edge_mask[src, tgt] > 0:
                valid_edges.append(i)
        
        if len(valid_edges) == 0:
            # Return self-loops if no valid edges
            num_nodes = len(self.ORGAN_INDICES)
            return torch.stack([
                torch.arange(num_nodes, device=device),
                torch.arange(num_nodes, device=device)
            ])
        
        return edge_index[:, valid_edges]
    
    def get_attention_weights(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability
        
        Returns:
            Attention weights [num_edges, num_heads]
        """
        # This would require modifying GAT to return attention weights
        # For now, return placeholder
        return torch.ones(edge_index.shape[1], self.num_heads)
    
    def predict_organ_state_change(
        self,
        current_state: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        time_delta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Predict change in organ states over time
        
        Args:
            current_state: Current organ feature values
            edge_index: Graph connectivity
            time_delta: Time step (months)
        
        Returns:
            Predicted state changes (delta)
        """
        # Forward pass to get refined states
        refined_states = self.forward(current_state, edge_index)
        
        # Calculate deltas
        deltas = {}
        for organ, features in refined_states.items():
            deltas[organ] = (features - current_state[organ]) * time_delta
        
        return deltas


class HybridOrganModel(nn.Module):
    """
    Hybrid model combining mechanistic equations with GNN learning
    
    Pipeline:
    1. Mechanistic simulation (physiological equations)
    2. GNN refinement (learned corrections)
    3. Combined prediction
    """
    
    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        mechanistic_model,  # PhysiologicalEquations instance
        hidden_dim: int = 64,
        learning_weight: float = 0.3
    ):
        """
        Initialize hybrid model
        
        Args:
            node_feature_dims: Feature dimensions per organ
            mechanistic_model: PhysiologicalEquations instance
            hidden_dim: GNN hidden dimension
            learning_weight: Weight for learned component (0-1)
                0 = pure mechanistic, 1 = pure learned
        """
        super(HybridOrganModel, self).__init__()
        
        self.mechanistic_model = mechanistic_model
        self.gnn = OrganGraphNetwork(
            node_feature_dims=node_feature_dims,
            hidden_dim=hidden_dim,
            use_physics_constraints=True
        )
        self.learning_weight = nn.Parameter(torch.tensor(learning_weight))
    
    def forward(
        self,
        patient_state: Dict,
        edge_index: torch.Tensor,
        time_delta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Hybrid prediction combining mechanistic and learned components
        
        Args:
            patient_state: Current patient state dictionary
            edge_index: Graph connectivity
            time_delta: Time step
        
        Returns:
            Predicted state changes
        """
        # 1. Mechanistic prediction
        mechanistic_deltas = self._mechanistic_prediction(patient_state, time_delta)
        
        # 2. GNN learned correction
        node_features = self._state_to_features(patient_state)
        learned_deltas = self.gnn.predict_organ_state_change(
            node_features, edge_index, time_delta
        )
        
        # 3. Combine predictions
        combined_deltas = {}
        weight = torch.sigmoid(self.learning_weight)  # Ensure 0-1
        
        for organ in mechanistic_deltas.keys():
            if organ in learned_deltas:
                combined_deltas[organ] = (
                    (1 - weight) * mechanistic_deltas[organ] +
                    weight * learned_deltas[organ]
                )
            else:
                combined_deltas[organ] = mechanistic_deltas[organ]
        
        return combined_deltas
    
    def _mechanistic_prediction(
        self,
        patient_state: Dict,
        time_delta: float
    ) -> Dict[str, torch.Tensor]:
        """Apply mechanistic physiological equations"""
        # This would call PhysiologicalEquations methods
        # Placeholder for now
        deltas = {}
        
        # Example: metabolic changes
        deltas['metabolic'] = torch.tensor([
            0.1,  # glucose change
            0.01,  # HbA1c change
            0.05,  # BMI change
            0.02   # waist change
        ]) * time_delta
        
        return deltas
    
    def _state_to_features(self, patient_state: Dict) -> Dict[str, torch.Tensor]:
        """Convert patient state to node features"""
        # Extract features for each organ system
        features = {}
        
        # This would use FeatureExtractor.extract_graph_features()
        # Placeholder for now
        features['metabolic'] = torch.tensor([
            patient_state.get('glucose', 100.0),
            patient_state.get('hba1c', 5.5),
            patient_state.get('bmi', 25.0),
            patient_state.get('waist', 90.0)
        ], dtype=torch.float32).unsqueeze(0)
        
        return features


def create_organ_graph_edges() -> torch.Tensor:
    """
    Create edge connectivity for organ graph
    
    Returns:
        edge_index tensor [2, num_edges]
    """
    edges = []
    
    # Add all known physiological edges
    for source, target in OrganGraphNetwork.KNOWN_EDGES:
        src_idx = OrganGraphNetwork.ORGAN_INDICES[source]
        tgt_idx = OrganGraphNetwork.ORGAN_INDICES[target]
        edges.append([src_idx, tgt_idx])
        # Bidirectional
        edges.append([tgt_idx, src_idx])
    
    # Add self-loops
    for i in range(len(OrganGraphNetwork.ORGAN_INDICES)):
        edges.append([i, i])
    
    return torch.tensor(edges, dtype=torch.long).t()


# Example usage
if __name__ == '__main__':
    # Define feature dimensions for each organ
    node_dims = {
        'metabolic': 4,  # glucose, HbA1c, BMI, waist
        'cardiovascular': 5,  # systolic, diastolic, LDL, HDL, TG
        'liver': 2,  # ALT, AST
        'kidney': 2,  # creatinine, eGFR
        'immune': 1,  # CRP
        'neural': 1,  # placeholder
        'lifestyle': 4  # exercise, smoking, alcohol, sleep
    }
    
    # Create model
    model = OrganGraphNetwork(
        node_feature_dims=node_dims,
        hidden_dim=64,
        num_attention_heads=4,
        num_layers=2
    )
    
    # Create example input
    node_features = {
        organ: torch.randn(1, dim)
        for organ, dim in node_dims.items()
    }
    
    # Create edges
    edge_index = create_organ_graph_edges()
    
    # Forward pass
    outputs = model(node_features, edge_index)
    
    print("Model created successfully")
    print(f"Input organs: {list(node_features.keys())}")
    print(f"Output organs: {list(outputs.keys())}")
    print(f"Edge index shape: {edge_index.shape}")
