#!/usr/bin/env python3
"""
Temporal Transformer Encoder for Patient Trajectory Modeling

Handles irregular time series data from NHANES with continuous time embeddings.
Learns long-range temporal dependencies across organ states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


class ContinuousTimeEmbedding(nn.Module):
    """
    Continuous time embedding for irregular time series
    Uses sinusoidal encoding similar to positional encoding but for continuous time
    """
    
    def __init__(self, d_model: int, max_time: float = 120.0):
        """
        Args:
            d_model: Embedding dimension
            max_time: Maximum time in months (default 120 = 10 years)
        """
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time
        
        # Learnable time scaling
        self.time_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_deltas: Time differences in months [batch_size, seq_len]
        
        Returns:
            Time embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = time_deltas.shape
        device = time_deltas.device
        
        # Normalize time to [0, 1]
        normalized_time = time_deltas / self.max_time * self.time_scale
        
        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() * 
            (-math.log(10000.0) / self.d_model)
        )
        
        # Sinusoidal encoding
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        pe[:, :, 0::2] = torch.sin(normalized_time.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(normalized_time.unsqueeze(-1) * div_term)
        
        return pe


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional masking"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for visualization
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # Store for visualization
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(out)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block"""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            mask: Attention mask
        
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Transformer for patient trajectory modeling
    
    Handles:
    - Irregular time series (continuous time embeddings)
    - Missing data (attention masking)
    - Long-range dependencies (multi-head attention)
    - Multi-organ interactions over time
    """
    
    def __init__(
        self,
        organ_embedding_dim: int = 64,
        num_organs: int = 7,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_time: float = 120.0
    ):
        """
        Args:
            organ_embedding_dim: Dimension of organ embeddings from GNN
            num_organs: Number of organ systems
            d_model: Transformer hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of Transformer blocks
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_time: Maximum time in months
        """
        super().__init__()
        
        self.organ_embedding_dim = organ_embedding_dim
        self.num_organs = num_organs
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Project organ embeddings to transformer dimension
        self.organ_projection = nn.Linear(organ_embedding_dim * num_organs, d_model)
        
        # Continuous time embedding
        self.time_embedding = ContinuousTimeEmbedding(d_model, max_time)
        
        # Learnable organ position embeddings
        self.organ_pos_embedding = nn.Parameter(
            torch.randn(1, num_organs, organ_embedding_dim)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        
        # Pooling for sequence aggregation
        self.pooling_type = 'attention'  # 'mean', 'max', or 'attention'
        if self.pooling_type == 'attention':
            self.attention_pool = nn.Linear(d_model, 1)
        
    def forward(
        self,
        organ_embeddings: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            organ_embeddings: [batch_size, seq_len, num_organs, organ_dim]
            time_deltas: Time in months from baseline [batch_size, seq_len]
            mask: Valid timestep mask [batch_size, seq_len]
        
        Returns:
            patient_embedding: [batch_size, d_model]
            attention_info: Dictionary with attention weights for visualization
        """
        batch_size, seq_len, num_organs, organ_dim = organ_embeddings.shape
        
        # Add organ position embeddings
        organ_embeddings = organ_embeddings + self.organ_pos_embedding
        
        # Flatten organs into single vector per timestep
        organ_flat = organ_embeddings.view(batch_size, seq_len, -1)
        
        # Project to transformer dimension
        x = self.organ_projection(organ_flat)  # [batch, seq_len, d_model]
        
        # Add time embeddings
        time_emb = self.time_embedding(time_deltas)
        x = x + time_emb
        
        # Create attention mask if needed
        if mask is not None:
            # Expand mask for attention: [batch, seq_len] -> [batch, seq_len, seq_len]
            attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        else:
            attn_mask = None
        
        # Pass through Transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x = block(x, attn_mask)
            if block.attention.attention_weights is not None:
                attention_weights.append(block.attention.attention_weights)
        
        # Final normalization
        x = self.output_norm(x)
        
        # Pool sequence to single patient embedding
        if self.pooling_type == 'mean':
            if mask is not None:
                # Masked mean
                x_masked = x * mask.unsqueeze(-1)
                patient_embedding = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                patient_embedding = x.mean(dim=1)
                
        elif self.pooling_type == 'max':
            patient_embedding = x.max(dim=1)[0]
            
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attn_scores = self.attention_pool(x).squeeze(-1)  # [batch, seq_len]
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len]
            patient_embedding = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        # Prepare attention info for visualization
        attention_info = {
            'layer_attention_weights': attention_weights,
            'pooling_weights': attn_weights if self.pooling_type == 'attention' else None,
            'time_deltas': time_deltas
        }
        
        return patient_embedding, attention_info
    
    def get_attention_maps(self, attention_info: Dict) -> Dict:
        """
        Extract attention maps for visualization
        
        Returns:
            Dictionary with averaged attention weights per layer
        """
        maps = {}
        
        for i, attn in enumerate(attention_info['layer_attention_weights']):
            # Average over heads: [batch, heads, seq, seq] -> [batch, seq, seq]
            maps[f'layer_{i}'] = attn.mean(dim=1)
        
        if attention_info['pooling_weights'] is not None:
            maps['pooling'] = attention_info['pooling_weights']
        
        return maps


class MaskedPretrainer(nn.Module):
    """
    Self-supervised pretraining with masked feature prediction
    
    Masks random features/timesteps and trains model to reconstruct them.
    Improves robustness to missing NHANES data.
    """
    
    def __init__(
        self,
        transformer: TemporalTransformerEncoder,
        organ_embedding_dim: int = 64,
        num_organs: int = 7,
        mask_prob: float = 0.15
    ):
        super().__init__()
        
        self.transformer = transformer
        self.organ_embedding_dim = organ_embedding_dim
        self.num_organs = num_organs
        self.mask_prob = mask_prob
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(transformer.d_model, transformer.d_model),
            nn.GELU(),
            nn.Linear(transformer.d_model, organ_embedding_dim * num_organs)
        )
        
        # Learnable mask token
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, num_organs, organ_embedding_dim)
        )
        
    def forward(
        self,
        organ_embeddings: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            organ_embeddings: [batch_size, seq_len, num_organs, organ_dim]
            time_deltas: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        
        Returns:
            loss: Reconstruction loss
            predictions: Reconstructed embeddings
            masked_positions: Which positions were masked
        """
        batch_size, seq_len, num_organs, organ_dim = organ_embeddings.shape
        
        # Create random mask
        masked_positions = torch.rand(batch_size, seq_len) < self.mask_prob
        masked_positions = masked_positions.to(organ_embeddings.device)
        
        # Don't mask invalid positions
        if mask is not None:
            masked_positions = masked_positions & mask.bool()
        
        # Replace masked positions with mask token
        masked_embeddings = organ_embeddings.clone()
        masked_embeddings[masked_positions] = self.mask_token
        
        # Forward through transformer
        patient_embedding, _ = self.transformer(
            masked_embeddings, time_deltas, mask
        )
        
        # Reconstruct masked features - output should match input shape
        # patient_embedding is [batch_size, d_model]
        # We need to reconstruct [batch_size, seq_len, num_organs, organ_dim]
        reconstructed = self.reconstruction_head(patient_embedding)
        reconstructed = reconstructed.view(batch_size, num_organs, organ_dim)
        
        # Calculate loss only on masked positions
        if masked_positions.any():
            # Flatten to [batch_size, seq_len, num_organs * organ_dim]
            original_flat = organ_embeddings.view(batch_size, seq_len, -1)
            
            # Average over sequence for each patient
            original_mean = original_flat.mean(dim=1)  # [batch_size, num_organs * organ_dim]
            
            # Flatten reconstructed
            reconstructed_flat = reconstructed.view(batch_size, -1)  # [batch_size, num_organs * organ_dim]
            
            # MSE loss between reconstructed and original mean
            loss = F.mse_loss(reconstructed_flat, original_mean)
        else:
            loss = torch.tensor(0.0, device=organ_embeddings.device, requires_grad=True)
        
        return loss, reconstructed, masked_positions


# Example usage
if __name__ == '__main__':
    # Test temporal transformer
    batch_size = 4
    seq_len = 12  # 12 months
    num_organs = 7
    organ_dim = 64
    
    # Create model
    model = TemporalTransformerEncoder(
        organ_embedding_dim=organ_dim,
        num_organs=num_organs,
        d_model=512,
        num_heads=8,
        num_layers=4
    )
    
    # Create dummy data
    organ_embeddings = torch.randn(batch_size, seq_len, num_organs, organ_dim)
    time_deltas = torch.arange(seq_len).float().unsqueeze(0).repeat(batch_size, 1)
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    patient_embedding, attention_info = model(organ_embeddings, time_deltas, mask)
    
    print(f"Input shape: {organ_embeddings.shape}")
    print(f"Patient embedding shape: {patient_embedding.shape}")
    print(f"Number of attention layers: {len(attention_info['layer_attention_weights'])}")
    
    # Test masked pretraining
    pretrainer = MaskedPretrainer(model, organ_dim, num_organs)
    loss, reconstructed, masked_pos = pretrainer(organ_embeddings, time_deltas, mask)
    
    print(f"\nPretraining loss: {loss.item():.4f}")
    print(f"Masked positions: {masked_pos.sum().item()}")
    print(f"Reconstructed shape: {reconstructed.shape}")
