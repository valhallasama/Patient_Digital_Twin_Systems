#!/usr/bin/env python3
"""
Run Stage 2 fine-tuning only, loading pretrained model from Stage 1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging
from train_two_stage import stage2_finetuning
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    config = {
        'nhanes_finetune_path': './data/nhanes_all_135310.pkl',
        'pretrained_model_path': './models/pretrained/pretrained_best.pt',
        'finetune_max_epochs': 150,
        'finetune_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("="*80)
    logger.info("STAGE 2 FINE-TUNING ONLY (Loading Pretrained Model)")
    logger.info("="*80)
    logger.info(f"Device: {config['device']}")
    
    # Create model with same architecture (dimensions must match data)
    node_dims = {
        'metabolic': 4,
        'cardiovascular': 5,
        'liver': 2,
        'kidney': 2,
        'immune': 1,
        'neural': 1,
        'lifestyle': 4  # Fixed: lifestyle has 4 features in the data
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
    ).to(config['device'])
    
    # Load pretrained GNN and Transformer weights from Stage 1
    logger.info(f"Loading pretrained model from {config['pretrained_model_path']}...")
    checkpoint = torch.load(config['pretrained_model_path'], map_location=config['device'])
    
    # Load GNN weights
    if 'gnn_state_dict' in checkpoint:
        model.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        logger.info(f"✓ Loaded pretrained GNN weights")
    else:
        logger.warning("⚠ GNN weights not found in checkpoint - using random initialization")
    
    # Load Transformer weights
    if 'transformer_state_dict' in checkpoint:
        model.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        logger.info(f"✓ Loaded pretrained Transformer weights")
    else:
        logger.warning("⚠ Transformer weights not found in checkpoint - using random initialization")
    
    logger.info(f"✓ Loaded pretrained model (Stage 1 val loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
    logger.info(f"  Prediction head and demographics encoder will be trained from scratch")
    
    # Stage 2: Fine-tune on 135K patients
    logger.info("="*80)
    logger.info("STAGE 2: SUPERVISED FINE-TUNING (135K patients)")
    logger.info("="*80)
    model = stage2_finetuning(
        model,
        data_path=config['nhanes_finetune_path'],
        max_epochs=config['finetune_max_epochs'],
        early_stop_patience=config['finetune_patience'],
        device=config['device']
    )
    
    logger.info("\n✓ Stage 2 fine-tuning complete!")

if __name__ == '__main__':
    main()
