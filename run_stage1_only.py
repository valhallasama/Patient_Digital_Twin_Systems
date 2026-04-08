#!/usr/bin/env python3
"""
Run Stage 1 pretraining only with proper GNN + Transformer saving
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging
from train_two_stage import stage1_pretraining
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    config = {
        'nhanes_pretrain_path': './data/nhanes_all_135310.pkl',
        'pretrain_max_epochs': 100,
        'pretrain_patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("="*80)
    logger.info("STAGE 1 PRETRAINING ONLY (135K patients)")
    logger.info("="*80)
    logger.info(f"Device: {config['device']}")
    
    # Create model (dimensions must match data)
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
    
    # Stage 1: Pretrain on 135K patients
    logger.info("="*80)
    logger.info("STAGE 1: SELF-SUPERVISED PRETRAINING (135K patients)")
    logger.info("="*80)
    model = stage1_pretraining(
        model,
        data_path=config['nhanes_pretrain_path'],
        max_epochs=config['pretrain_max_epochs'],
        early_stop_patience=config['pretrain_patience'],
        device=config['device']
    )
    
    logger.info("\n✓ Stage 1 pretraining complete!")
    logger.info("  GNN + Transformer weights saved to ./models/pretrained/pretrained_best.pt")
    logger.info("  Ready for Stage 2 fine-tuning")

if __name__ == '__main__':
    main()
