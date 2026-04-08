#!/usr/bin/env python3
"""
Training Pipeline for GNN-Transformer Hybrid Model

Trains on real NHANES data with:
1. Self-supervised pretraining (masked reconstruction)
2. Supervised multi-disease prediction
3. Evaluation and visualization
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid, HybridLoss
from graph_learning.temporal_transformer import MaskedPretrainer
from data_integration.nhanes_csv_loader import NHANESCSVLoader
from data_integration.feature_extractor import FeatureExtractor
from data_integration.comprehensive_disease_labels import ComprehensiveDiseaseLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHANESTemporalDataset(Dataset):
    """
    Dataset for temporal NHANES data
    
    Creates synthetic temporal sequences from cross-sectional NHANES data
    by simulating follow-up visits
    """
    
    def __init__(
        self,
        patients: List[Dict],
        seq_len: int = 12,
        time_horizon_months: int = 60
    ):
        self.patients = patients
        self.seq_len = seq_len
        self.time_horizon_months = time_horizon_months
        
        self.feature_extractor = FeatureExtractor()
        self.disease_labeler = ComprehensiveDiseaseLabeler()
        
        # Organ feature dimensions
        self.organ_dims = {
            'metabolic': 4,
            'cardiovascular': 5,
            'liver': 2,
            'kidney': 2,
            'immune': 1,
            'neural': 1,
            'lifestyle': 4
        }
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict:
        patient = self.patients[idx]
        
        # Extract graph features (baseline)
        graph_features = self.feature_extractor.extract_graph_features(patient)
        
        # Create temporal sequence by simulating progression
        # For now, use baseline + small random variations
        # In production, this would use actual longitudinal data or simulation
        organ_sequence = {}
        
        for organ, features in graph_features.items():
            # Create sequence with small variations
            seq = []
            for t in range(self.seq_len):
                # Add time-dependent noise (increases with time)
                noise = np.random.randn(len(features)) * 0.05 * (t / self.seq_len)
                seq.append(features + noise)
            organ_sequence[organ] = np.stack(seq)
        
        # Time deltas (months from baseline)
        time_deltas = np.linspace(0, self.time_horizon_months, self.seq_len)
        
        # Extract demographics
        demographics = np.array([
            patient.get('age', 50) / 100,  # Normalize
            1.0 if patient.get('sex') == 'male' else 0.0,
            patient.get('bmi', 25) / 50,
            patient.get('systolic_bp', 120) / 200,
            patient.get('diastolic_bp', 80) / 120,
            patient.get('fasting_glucose', 100) / 200,
            patient.get('hba1c', 5.5) / 15,
            patient.get('total_cholesterol', 200) / 400,
            1.0 if patient.get('smoking', False) else 0.0,
            patient.get('exercise_hours_per_week', 0) / 20
        ], dtype=np.float32)
        
        # Extract disease labels
        disease_labels_dict = self.disease_labeler.extract_all_disease_labels(patient)
        
        # Convert to binary array (24 diseases)
        disease_labels = np.array([
            float(disease_labels_dict.get(disease, False))
            for disease in GNNTransformerHybrid.DISEASE_NAMES
        ], dtype=np.float32)
        
        # Estimate time to onset (simplified)
        # For existing diseases: 0 months
        # For future diseases: random between 12-60 months
        time_to_onset = np.where(
            disease_labels > 0.5,
            0.0,  # Already has disease
            np.random.uniform(12, 60, size=len(disease_labels))
        ).astype(np.float32)
        
        return {
            'organ_features': organ_sequence,
            'time_deltas': time_deltas.astype(np.float32),
            'demographics': demographics,
            'disease_labels': disease_labels,
            'time_to_onset': time_to_onset,
            'patient_id': patient.get('patient_id', f'P{idx:06d}')
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    
    # Get organ names from first sample
    organ_names = list(batch[0]['organ_features'].keys())
    
    # Stack organ features
    organ_features = {}
    for organ in organ_names:
        organ_features[organ] = torch.FloatTensor(
            np.stack([sample['organ_features'][organ] for sample in batch])
        )
    
    # Stack other tensors
    time_deltas = torch.FloatTensor(
        np.stack([sample['time_deltas'] for sample in batch])
    )
    
    demographics = torch.FloatTensor(
        np.stack([sample['demographics'] for sample in batch])
    )
    
    disease_labels = torch.FloatTensor(
        np.stack([sample['disease_labels'] for sample in batch])
    )
    
    time_to_onset = torch.FloatTensor(
        np.stack([sample['time_to_onset'] for sample in batch])
    )
    
    patient_ids = [sample['patient_id'] for sample in batch]
    
    return {
        'organ_features': organ_features,
        'time_deltas': time_deltas,
        'demographics': demographics,
        'disease_labels': disease_labels,
        'time_to_onset': time_to_onset,
        'patient_ids': patient_ids
    }


def pretrain_model(
    model: GNNTransformerHybrid,
    train_loader: DataLoader,
    num_epochs: int = 10,
    device: str = 'cuda'
):
    """Self-supervised pretraining with masked reconstruction"""
    
    logger.info("Starting self-supervised pretraining...")
    
    # Create pretrainer
    pretrainer = MaskedPretrainer(
        transformer=model.transformer,
        organ_embedding_dim=model.gnn.hidden_dim,
        num_organs=model.num_organs,
        mask_prob=0.15
    ).to(device)
    
    optimizer = optim.AdamW(pretrainer.parameters(), lr=1e-4, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        pretrainer.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Move to device
            organ_features = {
                k: v.to(device) for k, v in batch['organ_features'].items()
            }
            time_deltas = batch['time_deltas'].to(device)
            
            # Apply GNN to get organ embeddings
            batch_size, seq_len = time_deltas.shape
            organ_embeddings_list = []
            
            for t in range(seq_len):
                timestep_features = {
                    organ: features[:, t, :] 
                    for organ, features in organ_features.items()
                }
                gnn_outputs = model.gnn(timestep_features, model.edge_index)
                organ_emb = torch.stack([
                    gnn_outputs[organ] 
                    for organ in sorted(model.node_feature_dims.keys())
                ], dim=1)
                organ_embeddings_list.append(organ_emb)
            
            organ_embeddings = torch.stack(organ_embeddings_list, dim=1)
            
            # Masked pretraining
            loss, _, _ = pretrainer(organ_embeddings, time_deltas)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    logger.info("Pretraining complete!")


def train_model(
    model: GNNTransformerHybrid,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    device: str = 'cuda',
    save_dir: str = './models/gnn_transformer'
):
    """Supervised training for multi-disease prediction"""
    
    logger.info("Starting supervised training...")
    
    model = model.to(device)
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_metrics = {'disease': 0, 'onset': 0, 'confidence': 0}
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Move to device
            organ_features = {
                k: v.to(device) for k, v in batch['organ_features'].items()
            }
            time_deltas = batch['time_deltas'].to(device)
            demographics = batch['demographics'].to(device)
            disease_labels = batch['disease_labels'].to(device)
            time_to_onset = batch['time_to_onset'].to(device)
            
            # Forward pass
            predictions, _ = model(organ_features, time_deltas, demographics)
            
            # Calculate loss
            targets = {
                'disease_labels': disease_labels,
                'time_to_onset': time_to_onset
            }
            
            loss, loss_components = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            for key in train_metrics:
                train_metrics[key] += loss_components[f'{key}_loss'].item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'disease': 0, 'onset': 0, 'confidence': 0}
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                organ_features = {
                    k: v.to(device) for k, v in batch['organ_features'].items()
                }
                time_deltas = batch['time_deltas'].to(device)
                demographics = batch['demographics'].to(device)
                disease_labels = batch['disease_labels'].to(device)
                time_to_onset = batch['time_to_onset'].to(device)
                
                predictions, _ = model(organ_features, time_deltas, demographics)
                
                targets = {
                    'disease_labels': disease_labels,
                    'time_to_onset': time_to_onset
                }
                
                loss, loss_components = criterion(predictions, targets)
                
                val_loss += loss.item()
                for key in val_metrics:
                    val_metrics[key] += loss_components[f'{key}_loss'].item()
                
                all_preds.append(predictions['risk_scores'].cpu().numpy())
                all_labels.append(disease_labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Calculate AUC scores
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        disease_aucs = {}
        for i, disease in enumerate(GNNTransformerHybrid.DISEASE_NAMES):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                disease_aucs[disease] = auc
        
        mean_auc = np.mean(list(disease_aucs.values()))
        
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Mean AUC: {mean_auc:.4f}")
        
        # Print top diseases by AUC
        top_diseases = sorted(disease_aucs.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 diseases by AUC:")
        for disease, auc in top_diseases:
            logger.info(f"  {disease}: {auc:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'mean_auc': mean_auc,
                'disease_aucs': disease_aucs
            }, save_path / 'best_model.pt')
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.4f})")
    
    logger.info("Training complete!")
    return disease_aucs


def evaluate_model(
    model: GNNTransformerHybrid,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict:
    """Comprehensive model evaluation"""
    
    logger.info("Evaluating model...")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            organ_features = {
                k: v.to(device) for k, v in batch['organ_features'].items()
            }
            time_deltas = batch['time_deltas'].to(device)
            demographics = batch['demographics'].to(device)
            
            predictions, _ = model(organ_features, time_deltas, demographics)
            
            all_preds.append(predictions['risk_scores'].cpu().numpy())
            all_labels.append(batch['disease_labels'].numpy())
            all_patient_ids.extend(batch['patient_ids'])
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics per disease
    results = {}
    
    for i, disease in enumerate(GNNTransformerHybrid.DISEASE_NAMES):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            acc = accuracy_score(all_labels[:, i], all_preds[:, i] > 0.5)
            
            results[disease] = {
                'auc': auc,
                'average_precision': ap,
                'accuracy': acc,
                'prevalence': all_labels[:, i].mean()
            }
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    for disease, metrics in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
        logger.info(f"\n{disease}:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  AP: {metrics['average_precision']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Prevalence: {metrics['prevalence']:.4f}")
    
    mean_auc = np.mean([m['auc'] for m in results.values()])
    logger.info(f"\nMean AUC across all diseases: {mean_auc:.4f}")
    
    return results


def main():
    # Configuration
    config = {
        'data_path': './data/nhanes_multi_disease_10k.pkl',
        'num_patients': 10000,
        'seq_len': 12,
        'batch_size': 32,
        'num_epochs_pretrain': 10,
        'num_epochs_train': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './models/gnn_transformer'
    }
    
    logger.info("="*80)
    logger.info("GNN-Transformer Hybrid Training Pipeline")
    logger.info("="*80)
    logger.info(f"Device: {config['device']}")
    
    # Load processed NHANES data
    logger.info(f"\nLoading data from {config['data_path']}...")
    
    if not Path(config['data_path']).exists():
        logger.error(f"Data file not found: {config['data_path']}")
        logger.info("Please run: python3 examples/process_nhanes_multi_disease.py")
        return
    
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    
    patients = data['patients'][:config['num_patients']]
    logger.info(f"Loaded {len(patients)} patients")
    
    # Split data
    train_size = int(0.7 * len(patients))
    val_size = int(0.15 * len(patients))
    
    train_patients = patients[:train_size]
    val_patients = patients[train_size:train_size+val_size]
    test_patients = patients[train_size+val_size:]
    
    logger.info(f"Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    # Create datasets
    train_dataset = NHANESTemporalDataset(train_patients, seq_len=config['seq_len'])
    val_dataset = NHANESTemporalDataset(val_patients, seq_len=config['seq_len'])
    test_dataset = NHANESTemporalDataset(test_patients, seq_len=config['seq_len'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    logger.info("\nCreating GNN-Transformer Hybrid model...")
    
    node_dims = train_dataset.organ_dims
    
    model = GNNTransformerHybrid(
        node_feature_dims=node_dims,
        gnn_hidden_dim=64,
        transformer_d_model=512,
        transformer_num_heads=8,
        transformer_num_layers=4,
        num_diseases=24,
        use_demographics=True,
        demographic_dim=10
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Pretraining
    if config['num_epochs_pretrain'] > 0:
        pretrain_model(
            model,
            train_loader,
            num_epochs=config['num_epochs_pretrain'],
            device=config['device']
        )
    
    # Training
    disease_aucs = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs_train'],
        device=config['device'],
        save_dir=config['save_dir']
    )
    
    # Load best model
    checkpoint = torch.load(Path(config['save_dir']) / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluation
    test_results = evaluate_model(model, test_loader, device=config['device'])
    
    # Save results
    results_path = Path(config['save_dir']) / 'evaluation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(test_results, f)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
