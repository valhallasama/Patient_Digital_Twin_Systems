#!/usr/bin/env python3
"""
Attention Visualization Tools

Visualize attention patterns from GNN-Transformer model for interpretability.
Shows which organs and time points are most important for disease predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


class AttentionVisualizer:
    """Visualize attention weights from Transformer model"""
    
    ORGAN_NAMES = [
        'Metabolic', 'Cardiovascular', 'Liver', 
        'Kidney', 'Immune', 'Neural', 'Lifestyle'
    ]
    
    def __init__(self, save_dir: str = './outputs/attention_maps'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_temporal_attention(
        self,
        attention_weights: torch.Tensor,
        time_deltas: np.ndarray,
        disease_name: str,
        patient_id: str = 'P00001',
        save: bool = True
    ):
        """
        Plot temporal attention heatmap
        
        Args:
            attention_weights: [seq_len, seq_len] attention matrix
            time_deltas: Time points in months
            disease_name: Name of disease being predicted
            patient_id: Patient identifier
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Plot heatmap
        sns.heatmap(
            attention_weights,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=[f'{int(t)}m' for t in time_deltas],
            yticklabels=[f'{int(t)}m' for t in time_deltas],
            ax=ax,
            vmin=0,
            vmax=attention_weights.max()
        )
        
        ax.set_xlabel('Time Point (months)', fontsize=12)
        ax.set_ylabel('Time Point (months)', fontsize=12)
        ax.set_title(
            f'Temporal Attention Pattern\n{disease_name} - Patient {patient_id}',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save:
            filename = f'temporal_attention_{disease_name}_{patient_id}.png'
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_organ_importance(
        self,
        attention_weights: np.ndarray,
        disease_name: str,
        top_k: int = 7,
        save: bool = True
    ):
        """
        Plot organ importance for disease prediction
        
        Args:
            attention_weights: [num_organs] importance scores
            disease_name: Name of disease
            top_k: Number of top organs to show
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort organs by importance
        organ_scores = list(zip(self.ORGAN_NAMES, attention_weights))
        organ_scores.sort(key=lambda x: x[1], reverse=True)
        
        organs = [x[0] for x in organ_scores[:top_k]]
        scores = [x[1] for x in organ_scores[:top_k]]
        
        # Create bar plot
        colors = sns.color_palette('viridis', len(organs))
        bars = ax.barh(organs, scores, color=colors)
        
        # Add value labels
        for i, (organ, score) in enumerate(zip(organs, scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', 
                   va='center', fontsize=10)
        
        ax.set_xlabel('Attention Weight', fontsize=12)
        ax.set_ylabel('Organ System', fontsize=12)
        ax.set_title(
            f'Organ System Importance for {disease_name}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlim(0, max(scores) * 1.15)
        
        plt.tight_layout()
        
        if save:
            filename = f'organ_importance_{disease_name}.png'
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_attention_evolution(
        self,
        layer_attentions: List[torch.Tensor],
        time_deltas: np.ndarray,
        patient_id: str = 'P00001',
        save: bool = True
    ):
        """
        Plot how attention evolves across Transformer layers
        
        Args:
            layer_attentions: List of [seq_len, seq_len] attention matrices
            time_deltas: Time points
            patient_id: Patient identifier
            save: Whether to save
        """
        num_layers = len(layer_attentions)
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
        
        if num_layers == 1:
            axes = [axes]
        
        for i, (ax, attn) in enumerate(zip(axes, layer_attentions)):
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().numpy()
            
            sns.heatmap(
                attn,
                cmap='YlOrRd',
                cbar=True,
                xticklabels=[f'{int(t)}m' for t in time_deltas],
                yticklabels=[f'{int(t)}m' for t in time_deltas],
                ax=ax,
                vmin=0,
                vmax=1.0
            )
            
            ax.set_title(f'Layer {i+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (months)')
            if i == 0:
                ax.set_ylabel('Time (months)')
        
        plt.suptitle(
            f'Attention Evolution Across Layers - Patient {patient_id}',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()
        
        if save:
            filename = f'attention_evolution_{patient_id}.png'
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_pooling_attention(
        self,
        pooling_weights: np.ndarray,
        time_deltas: np.ndarray,
        disease_name: str,
        patient_id: str = 'P00001',
        save: bool = True
    ):
        """
        Plot attention-based pooling weights
        Shows which time points are most important for final prediction
        
        Args:
            pooling_weights: [seq_len] pooling attention weights
            time_deltas: Time points
            disease_name: Disease being predicted
            patient_id: Patient identifier
            save: Whether to save
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create bar plot
        colors = plt.cm.viridis(pooling_weights / pooling_weights.max())
        bars = ax.bar(time_deltas, pooling_weights, color=colors, width=3)
        
        # Add value labels
        for t, w in zip(time_deltas, pooling_weights):
            if w > pooling_weights.max() * 0.1:  # Only label significant weights
                ax.text(t, w + 0.01, f'{w:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Time (months from baseline)', fontsize=12)
        ax.set_ylabel('Pooling Attention Weight', fontsize=12)
        ax.set_title(
            f'Temporal Importance for {disease_name}\nPatient {patient_id}',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f'pooling_attention_{disease_name}_{patient_id}.png'
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def plot_multi_disease_attention(
        self,
        disease_attentions: Dict[str, np.ndarray],
        time_deltas: np.ndarray,
        patient_id: str = 'P00001',
        top_diseases: int = 6,
        save: bool = True
    ):
        """
        Compare attention patterns across multiple diseases
        
        Args:
            disease_attentions: Dict mapping disease name to pooling weights
            time_deltas: Time points
            patient_id: Patient identifier
            top_diseases: Number of diseases to show
            save: Whether to save
        """
        # Select top diseases by max attention
        sorted_diseases = sorted(
            disease_attentions.items(),
            key=lambda x: x[1].max(),
            reverse=True
        )[:top_diseases]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, (disease, weights) in enumerate(sorted_diseases):
            ax = axes[i]
            
            colors = plt.cm.viridis(weights / weights.max())
            ax.bar(time_deltas, weights, color=colors, width=3)
            
            ax.set_title(disease.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Months', fontsize=9)
            ax.set_ylabel('Attention', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(sorted_diseases), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(
            f'Disease-Specific Temporal Attention Patterns\nPatient {patient_id}',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        
        if save:
            filename = f'multi_disease_attention_{patient_id}.png'
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def create_attention_report(
        self,
        attention_info: Dict,
        predictions: Dict,
        patient_id: str = 'P00001',
        save: bool = True
    ):
        """
        Create comprehensive attention visualization report
        
        Args:
            attention_info: Attention information from model
            predictions: Disease predictions
            patient_id: Patient identifier
            save: Whether to save
        """
        print(f"\nGenerating attention report for patient {patient_id}...")
        
        time_deltas = attention_info['time_deltas'].cpu().numpy()[0]
        
        # 1. Temporal attention (first layer)
        if len(attention_info['layer_attention_weights']) > 0:
            first_layer_attn = attention_info['layer_attention_weights'][0][0].mean(0)
            self.plot_temporal_attention(
                first_layer_attn,
                time_deltas,
                'Overall',
                patient_id,
                save
            )
        
        # 2. Attention evolution across layers
        if len(attention_info['layer_attention_weights']) > 1:
            layer_attns = [
                attn[0].mean(0) for attn in attention_info['layer_attention_weights']
            ]
            self.plot_attention_evolution(
                layer_attns,
                time_deltas,
                patient_id,
                save
            )
        
        # 3. Pooling attention
        if attention_info['pooling_weights'] is not None:
            pooling = attention_info['pooling_weights'].cpu().numpy()[0]
            self.plot_pooling_attention(
                pooling,
                time_deltas,
                'Overall',
                patient_id,
                save
            )
        
        print(f"Attention report complete for patient {patient_id}")


# Example usage
if __name__ == '__main__':
    visualizer = AttentionVisualizer()
    
    # Create dummy data
    seq_len = 12
    time_deltas = np.arange(0, 60, 5)
    
    # Temporal attention
    attention = np.random.rand(seq_len, seq_len)
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    visualizer.plot_temporal_attention(
        torch.tensor(attention),
        time_deltas,
        'Diabetes',
        'P00001'
    )
    
    # Organ importance
    organ_weights = np.random.rand(7)
    organ_weights = organ_weights / organ_weights.sum()
    
    visualizer.plot_organ_importance(
        organ_weights,
        'Diabetes'
    )
    
    # Pooling attention
    pooling = np.random.rand(seq_len)
    pooling = pooling / pooling.sum()
    
    visualizer.plot_pooling_attention(
        pooling,
        time_deltas,
        'Diabetes',
        'P00001'
    )
    
    print("Visualization examples created!")
