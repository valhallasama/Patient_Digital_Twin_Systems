#!/usr/bin/env python3
"""
GNN-Transformer Hybrid Model for Multi-Disease Prediction

Combines:
1. OrganGraphNetwork (spatial organ interactions)
2. TemporalTransformerEncoder (temporal dependencies)
3. Multi-disease prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .organ_gnn import OrganGraphNetwork, create_organ_graph_edges
from .temporal_transformer import TemporalTransformerEncoder


class MultiDiseasePredictionHead(nn.Module):
    """
    Multi-task prediction head for 24 diseases
    
    Predicts:
    - Disease risk (0-1 probability)
    - Time to onset (months)
    - Confidence score
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_diseases: int = 24,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_diseases = num_diseases
        
        # Shared representation
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Disease-specific heads
        self.risk_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_diseases)
        ])
        
        self.onset_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_diseases)
        ])
        
        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_dim, num_diseases)
        
    def forward(self, patient_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            patient_embedding: [batch_size, input_dim]
        
        Returns:
            Dictionary with predictions for each disease
        """
        batch_size = patient_embedding.shape[0]
        
        # Shared representation
        shared = self.shared_layers(patient_embedding)
        
        # Disease risk predictions
        risks = []
        for head in self.risk_heads:
            risk = torch.sigmoid(head(shared))
            risks.append(risk)
        risks = torch.cat(risks, dim=1)  # [batch, num_diseases]
        
        # Time to onset predictions (in months, positive values)
        onsets = []
        for head in self.onset_heads:
            onset = F.softplus(head(shared))  # Ensure positive
            onsets.append(onset)
        onsets = torch.cat(onsets, dim=1)  # [batch, num_diseases]
        
        # Confidence scores
        confidence = torch.sigmoid(self.confidence_head(shared))
        
        return {
            'risk_scores': risks,
            'time_to_onset': onsets,
            'confidence': confidence
        }


class GNNTransformerHybrid(nn.Module):
    """
    Complete hybrid model combining GNN and Transformer
    
    Architecture:
    1. Per-timestep GNN: Learn organ interactions (spatial)
    2. Temporal Transformer: Learn trajectory patterns (temporal)
    3. Multi-disease heads: Predict 24 diseases
    """
    
    # Disease names (24 total)
    DISEASE_NAMES = [
        # Metabolic (4)
        'diabetes', 'prediabetes', 'metabolic_syndrome', 'obesity',
        # Cardiovascular (7)
        'hypertension', 'prehypertension', 'coronary_heart_disease',
        'heart_failure', 'stroke', 'dyslipidemia', 'high_cvd_risk',
        # Kidney (4)
        'chronic_kidney_disease', 'ckd_stage_3', 'ckd_stage_4', 'ckd_stage_5',
        # Liver (3)
        'nafld', 'elevated_liver_enzymes', 'liver_disease',
        # Respiratory (2)
        'copd', 'asthma',
        # Hematologic (1)
        'anemia',
        # Endocrine (1)
        'hypothyroidism',
        # Inflammatory (1)
        'chronic_inflammation',
        # Oncologic (1)
        'cancer_any'
    ]
    
    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        gnn_hidden_dim: int = 64,
        transformer_d_model: int = 512,
        transformer_num_heads: int = 8,
        transformer_num_layers: int = 4,
        num_diseases: int = 24,
        use_demographics: bool = True,
        demographic_dim: int = 10
    ):
        """
        Args:
            node_feature_dims: Feature dimensions per organ system
            gnn_hidden_dim: GNN hidden dimension
            transformer_d_model: Transformer hidden dimension
            transformer_num_heads: Number of attention heads
            transformer_num_layers: Number of Transformer blocks
            num_diseases: Number of diseases to predict
            use_demographics: Whether to include demographics
            demographic_dim: Dimension of demographic features
        """
        super().__init__()
        
        self.node_feature_dims = node_feature_dims
        self.num_organs = len(node_feature_dims)
        self.use_demographics = use_demographics
        
        # 1. Organ GNN (per timestep)
        self.gnn = OrganGraphNetwork(
            node_feature_dims=node_feature_dims,
            hidden_dim=gnn_hidden_dim,
            num_attention_heads=4,
            num_layers=2,
            use_physics_constraints=True
        )
        
        # 2. Temporal Transformer
        self.transformer = TemporalTransformerEncoder(
            organ_embedding_dim=gnn_hidden_dim,
            num_organs=self.num_organs,
            d_model=transformer_d_model,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers
        )
        
        # 3. Demographics encoder (optional)
        if use_demographics:
            self.demographic_encoder = nn.Sequential(
                nn.Linear(demographic_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 128)
            )
            combined_dim = transformer_d_model + 128
        else:
            combined_dim = transformer_d_model
        
        # 4. Multi-disease prediction heads
        self.prediction_head = MultiDiseasePredictionHead(
            input_dim=combined_dim,
            num_diseases=num_diseases
        )
        
        # Store edge index for GNN
        self.register_buffer('edge_index', create_organ_graph_edges())
        
    def forward(
        self,
        organ_features_sequence: Dict[str, torch.Tensor],
        time_deltas: torch.Tensor,
        demographics: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Args:
            organ_features_sequence: Dictionary mapping organ name to features
                Each tensor: [batch_size, seq_len, feature_dim]
            time_deltas: Time in months from baseline [batch_size, seq_len]
            demographics: Patient demographics [batch_size, demographic_dim]
            mask: Valid timestep mask [batch_size, seq_len]
        
        Returns:
            predictions: Dictionary with disease predictions
            attention_info: Attention weights for visualization
        """
        batch_size = time_deltas.shape[0]
        seq_len = time_deltas.shape[1]
        
        # Step 1: Apply GNN to each timestep
        organ_embeddings_list = []
        
        for t in range(seq_len):
            # Extract features at timestep t
            timestep_features = {
                organ: features[:, t, :] 
                for organ, features in organ_features_sequence.items()
            }
            
            # Apply GNN (return hidden embeddings, not projected back to original dims)
            gnn_outputs = self.gnn(timestep_features, self.edge_index, return_hidden=True)
            
            # Stack organ embeddings (all have same hidden_dim now)
            organ_emb = torch.stack([
                gnn_outputs[organ] 
                for organ in sorted(self.node_feature_dims.keys())
            ], dim=1)  # [batch, num_organs, gnn_hidden_dim]
            
            organ_embeddings_list.append(organ_emb)
        
        # Stack all timesteps
        organ_embeddings = torch.stack(organ_embeddings_list, dim=1)
        # [batch, seq_len, num_organs, gnn_hidden_dim]
        
        # Step 2: Apply Temporal Transformer
        patient_embedding, attention_info = self.transformer(
            organ_embeddings, time_deltas, mask
        )
        # [batch, transformer_d_model]
        
        # Step 3: Combine with demographics if available
        if self.use_demographics and demographics is not None:
            demo_emb = self.demographic_encoder(demographics)
            combined_embedding = torch.cat([patient_embedding, demo_emb], dim=1)
        else:
            combined_embedding = patient_embedding
        
        # Step 4: Multi-disease prediction
        predictions = self.prediction_head(combined_embedding)
        
        return predictions, attention_info
    
    def predict_diseases(
        self,
        organ_features_sequence: Dict[str, torch.Tensor],
        time_deltas: torch.Tensor,
        demographics: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Predict diseases with interpretable output
        
        Returns:
            List of dictionaries (one per patient) with disease predictions
        """
        self.eval()
        with torch.no_grad():
            predictions, attention_info = self.forward(
                organ_features_sequence, time_deltas, demographics, mask
            )
        
        batch_size = predictions['risk_scores'].shape[0]
        results = []
        
        for i in range(batch_size):
            patient_result = {
                'diseases': {},
                'overall_risk': predictions['risk_scores'][i].mean().item()
            }
            
            for j, disease_name in enumerate(self.DISEASE_NAMES):
                risk = predictions['risk_scores'][i, j].item()
                onset = predictions['time_to_onset'][i, j].item()
                conf = predictions['confidence'][i, j].item()
                
                patient_result['diseases'][disease_name] = {
                    'risk_score': risk,
                    'risk_percentage': risk * 100,
                    'predicted': risk >= threshold,
                    'time_to_onset_months': onset,
                    'time_to_onset_years': onset / 12,
                    'confidence': conf,
                    'risk_level': self._get_risk_level(risk)
                }
            
            results.append(patient_result)
        
        return results
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level"""
        if risk_score < 0.2:
            return "very_low"
        elif risk_score < 0.4:
            return "low"
        elif risk_score < 0.6:
            return "moderate"
        elif risk_score < 0.8:
            return "high"
        else:
            return "very_high"


class HybridLoss(nn.Module):
    """
    Multi-task loss for hybrid model
    
    Combines:
    - Binary cross-entropy for disease classification
    - MSE for time-to-onset regression
    - Confidence calibration loss
    """
    
    def __init__(
        self,
        disease_weights: Optional[torch.Tensor] = None,
        onset_weight: float = 0.3,
        confidence_weight: float = 0.1
    ):
        super().__init__()
        
        self.disease_weights = disease_weights
        self.onset_weight = onset_weight
        self.confidence_weight = confidence_weight
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: Model predictions
            targets: Ground truth labels
        
        Returns:
            total_loss: Combined loss
            loss_components: Individual loss values
        """
        # Disease classification loss
        disease_loss = F.binary_cross_entropy(
            predictions['risk_scores'],
            targets['disease_labels'],
            weight=self.disease_weights
        )
        
        # Time-to-onset loss (only for positive cases)
        if 'time_to_onset' in targets:
            positive_mask = targets['disease_labels'] > 0.5
            if positive_mask.any():
                onset_loss = F.mse_loss(
                    predictions['time_to_onset'][positive_mask],
                    targets['time_to_onset'][positive_mask]
                )
            else:
                onset_loss = torch.tensor(0.0, device=disease_loss.device)
        else:
            onset_loss = torch.tensor(0.0, device=disease_loss.device)
        
        # Confidence calibration loss
        # Confidence should match prediction accuracy
        pred_correct = (
            (predictions['risk_scores'] > 0.5) == 
            (targets['disease_labels'] > 0.5)
        ).float()
        confidence_loss = F.mse_loss(
            predictions['confidence'],
            pred_correct
        )
        
        # Total loss
        total_loss = (
            disease_loss + 
            self.onset_weight * onset_loss + 
            self.confidence_weight * confidence_loss
        )
        
        loss_components = {
            'disease_loss': disease_loss,
            'onset_loss': onset_loss,
            'confidence_loss': confidence_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components


# Example usage
if __name__ == '__main__':
    # Define organ feature dimensions
    node_dims = {
        'metabolic': 4,
        'cardiovascular': 5,
        'liver': 2,
        'kidney': 2,
        'immune': 1,
        'neural': 1,
        'lifestyle': 4
    }
    
    # Create model
    model = GNNTransformerHybrid(
        node_feature_dims=node_dims,
        gnn_hidden_dim=64,
        transformer_d_model=512,
        transformer_num_heads=8,
        transformer_num_layers=4,
        num_diseases=24
    )
    
    # Create dummy data
    batch_size = 4
    seq_len = 12
    
    organ_features = {
        organ: torch.randn(batch_size, seq_len, dim)
        for organ, dim in node_dims.items()
    }
    
    time_deltas = torch.arange(seq_len).float().unsqueeze(0).repeat(batch_size, 1)
    demographics = torch.randn(batch_size, 10)
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    predictions, attention_info = model(
        organ_features, time_deltas, demographics, mask
    )
    
    print("Model created successfully!")
    print(f"Risk scores shape: {predictions['risk_scores'].shape}")
    print(f"Time to onset shape: {predictions['time_to_onset'].shape}")
    print(f"Confidence shape: {predictions['confidence'].shape}")
    
    # Test prediction
    results = model.predict_diseases(organ_features, time_deltas, demographics, mask)
    print(f"\nPredicted {len(results[0]['diseases'])} diseases for {len(results)} patients")
    print(f"Overall risk: {results[0]['overall_risk']:.3f}")
