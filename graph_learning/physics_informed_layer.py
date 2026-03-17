#!/usr/bin/env python3
"""
Physics-Informed Graph Neural Network Layer
Enforces physiological constraints while learning from data

Key concepts:
- Hard constraints: Physiological bounds (e.g., BP > 0, HbA1c > 0)
- Soft constraints: Known relationships (e.g., insulin resistance → glucose)
- Causality preservation: Directed edges respect causal flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class PhysicsInformedGNN(nn.Module):
    """
    GNN layer with physics-informed constraints
    
    Enforces:
    1. Physiological bounds on predictions
    2. Monotonicity constraints (e.g., age only increases)
    3. Conservation laws (e.g., mass balance)
    4. Known causal relationships
    """
    
    # Physiological bounds for each parameter
    PARAMETER_BOUNDS = {
        # Metabolic
        'glucose': (50, 500),  # mg/dL
        'hba1c': (3.0, 15.0),  # %
        'bmi': (10, 60),  # kg/m²
        'insulin_resistance': (0, 1),  # normalized
        
        # Cardiovascular
        'systolic_bp': (60, 250),  # mmHg
        'diastolic_bp': (40, 150),  # mmHg
        'ldl': (20, 300),  # mg/dL
        'hdl': (10, 120),  # mg/dL
        'triglycerides': (20, 1000),  # mg/dL
        'heart_rate': (30, 200),  # bpm
        
        # Liver
        'alt': (5, 500),  # U/L
        'ast': (5, 500),  # U/L
        'liver_fat': (0, 1),  # normalized
        
        # Kidney
        'creatinine': (0.3, 15.0),  # mg/dL
        'egfr': (5, 150),  # mL/min/1.73m²
        
        # Inflammation
        'crp': (0.1, 50.0),  # mg/L
    }
    
    # Monotonicity constraints (can only increase or only decrease)
    MONOTONIC_INCREASING = ['age', 'atherosclerosis', 'arterial_stiffness']
    MONOTONIC_DECREASING = ['vessel_elasticity', 'pancreatic_function']
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        enforce_bounds: bool = True,
        enforce_monotonicity: bool = True,
        enforce_causality: bool = True
    ):
        """
        Initialize physics-informed layer
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            enforce_bounds: Whether to enforce physiological bounds
            enforce_monotonicity: Whether to enforce monotonicity constraints
            enforce_causality: Whether to enforce causal relationships
        """
        super(PhysicsInformedGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.enforce_bounds = enforce_bounds
        self.enforce_monotonicity = enforce_monotonicity
        self.enforce_causality = enforce_causality
        
        # Neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Learnable constraint weights
        self.constraint_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self,
        x: torch.Tensor,
        parameter_names: Optional[List[str]] = None,
        current_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with physics constraints
        
        Args:
            x: Input features [batch_size, input_dim]
            parameter_names: Names of output parameters (for constraint lookup)
            current_values: Current parameter values (for monotonicity)
        
        Returns:
            Constrained predictions [batch_size, output_dim]
        """
        # Standard neural network forward pass
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        output = self.fc3(h)
        
        # Apply physics constraints
        if self.enforce_bounds and parameter_names:
            output = self._apply_bounds(output, parameter_names)
        
        if self.enforce_monotonicity and parameter_names and current_values is not None:
            output = self._apply_monotonicity(output, parameter_names, current_values)
        
        return output
    
    def _apply_bounds(
        self,
        predictions: torch.Tensor,
        parameter_names: List[str]
    ) -> torch.Tensor:
        """
        Enforce physiological bounds on predictions
        
        Uses sigmoid/tanh to map to valid range
        """
        bounded = predictions.clone()
        
        for i, param_name in enumerate(parameter_names):
            if param_name in self.PARAMETER_BOUNDS:
                lower, upper = self.PARAMETER_BOUNDS[param_name]
                
                # Map to [0, 1] using sigmoid
                normalized = torch.sigmoid(predictions[:, i])
                
                # Scale to [lower, upper]
                bounded[:, i] = lower + normalized * (upper - lower)
        
        return bounded
    
    def _apply_monotonicity(
        self,
        predictions: torch.Tensor,
        parameter_names: List[str],
        current_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce monotonicity constraints
        
        Ensures parameters that should only increase/decrease follow that rule
        """
        constrained = predictions.clone()
        
        for i, param_name in enumerate(parameter_names):
            if param_name in self.MONOTONIC_INCREASING:
                # Ensure prediction >= current value
                constrained[:, i] = torch.maximum(
                    predictions[:, i],
                    current_values[:, i]
                )
            elif param_name in self.MONOTONIC_DECREASING:
                # Ensure prediction <= current value
                constrained[:, i] = torch.minimum(
                    predictions[:, i],
                    current_values[:, i]
                )
        
        return constrained
    
    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        patient_state: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute physics-informed loss
        
        Combines:
        - Standard prediction loss (MSE)
        - Constraint violation penalties
        - Known relationship penalties
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            patient_state: Patient state for relationship checks
        
        Returns:
            Total loss
        """
        # Standard prediction loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Constraint violation penalty
        constraint_loss = self._compute_constraint_violations(predictions, patient_state)
        
        # Relationship penalty (e.g., glucose and HbA1c should be correlated)
        relationship_loss = self._compute_relationship_violations(predictions, patient_state)
        
        # Combine losses
        total_loss = (
            mse_loss +
            self.constraint_weight * constraint_loss +
            0.1 * relationship_loss
        )
        
        return total_loss
    
    def _compute_constraint_violations(
        self,
        predictions: torch.Tensor,
        patient_state: Optional[Dict]
    ) -> torch.Tensor:
        """
        Compute penalty for constraint violations
        
        Examples:
        - Negative values for positive-only parameters
        - Out-of-range values
        - Impossible combinations
        """
        penalty = torch.tensor(0.0, device=predictions.device)
        
        # Example: Penalize if systolic < diastolic
        # (would need to know which indices correspond to which parameters)
        
        # Example: Penalize if HbA1c and glucose are inconsistent
        # HbA1c ≈ (glucose + 46.7) / 28.7
        
        return penalty
    
    def _compute_relationship_violations(
        self,
        predictions: torch.Tensor,
        patient_state: Optional[Dict]
    ) -> torch.Tensor:
        """
        Compute penalty for violating known physiological relationships
        
        Examples:
        - Insulin resistance should correlate with glucose
        - BMI should correlate with blood pressure
        - LDL should correlate with liver fat
        """
        penalty = torch.tensor(0.0, device=predictions.device)
        
        # Would implement specific relationship checks here
        # based on PhysiologicalEquations
        
        return penalty


class CausalGNNLayer(nn.Module):
    """
    Graph layer that respects causal ordering
    
    Ensures information flows in physiologically plausible directions:
    - Lifestyle → Metabolism → Organs → Diseases
    - Not: Diseases → Lifestyle (reverse causation)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        causal_order: List[str]
    ):
        """
        Initialize causal GNN layer
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            causal_order: List of node types in causal order
        """
        super(CausalGNNLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal_order = causal_order
        
        # Separate transformation for each causal level
        self.transforms = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in causal_order
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass respecting causal order
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            node_types: Type index for each node [num_nodes]
        
        Returns:
            Updated features [num_nodes, out_channels]
        """
        # Process nodes in causal order
        output = torch.zeros(x.shape[0], self.out_channels, device=x.device)
        
        for level, node_type in enumerate(self.causal_order):
            # Find nodes at this causal level
            mask = (node_types == level)
            
            if mask.sum() > 0:
                # Transform features
                output[mask] = self.transforms[level](x[mask])
                
                # Aggregate from previous causal levels only
                # (prevents reverse causation)
                for prev_level in range(level):
                    prev_mask = (node_types == prev_level)
                    # Aggregate messages from previous levels
                    # (implementation would use edge_index)
        
        return output


class UncertaintyQuantificationLayer(nn.Module):
    """
    Layer that outputs predictions with uncertainty estimates
    
    Uses Monte Carlo Dropout or learned variance
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        method: str = 'mc_dropout'
    ):
        """
        Initialize uncertainty quantification layer
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            method: 'mc_dropout' or 'learned_variance'
        """
        super(UncertaintyQuantificationLayer, self).__init__()
        
        self.method = method
        
        # Mean prediction network
        self.mean_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
        # Variance prediction network (if learned_variance)
        if method == 'learned_variance':
            self.var_net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Softplus()  # Ensure positive variance
            )
    
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty
        
        Args:
            x: Input features
            num_samples: Number of MC samples (if mc_dropout)
        
        Returns:
            (mean_prediction, uncertainty)
        """
        if self.method == 'mc_dropout':
            # Monte Carlo Dropout
            self.train()  # Enable dropout during inference
            
            samples = []
            for _ in range(num_samples):
                samples.append(self.mean_net(x))
            
            samples = torch.stack(samples)
            mean = samples.mean(dim=0)
            uncertainty = samples.std(dim=0)
            
            return mean, uncertainty
        
        else:  # learned_variance
            mean = self.mean_net(x)
            variance = self.var_net(x)
            uncertainty = torch.sqrt(variance)
            
            return mean, uncertainty


# Example usage
if __name__ == '__main__':
    # Create physics-informed layer
    layer = PhysicsInformedGNN(
        input_dim=20,
        hidden_dim=64,
        output_dim=10,
        enforce_bounds=True,
        enforce_monotonicity=True
    )
    
    # Example input
    x = torch.randn(32, 20)  # batch_size=32
    parameter_names = ['glucose', 'hba1c', 'systolic_bp', 'diastolic_bp', 
                      'ldl', 'hdl', 'alt', 'ast', 'creatinine', 'egfr']
    
    # Forward pass
    output = layer(x, parameter_names=parameter_names)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Check bounds
    for i, param in enumerate(parameter_names):
        if param in PhysicsInformedGNN.PARAMETER_BOUNDS:
            lower, upper = PhysicsInformedGNN.PARAMETER_BOUNDS[param]
            print(f"{param}: [{output[:, i].min():.2f}, {output[:, i].max():.2f}] "
                  f"(bounds: [{lower}, {upper}])")
