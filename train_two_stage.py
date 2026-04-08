#!/usr/bin/env python3
"""
Two-Stage Training Pipeline for GNN-Transformer

Stage 1: Self-supervised pretraining on FULL 135K dataset (all patients, even incomplete)
Stage 2: Supervised fine-tuning on 67K complete dataset with rare disease handling

This maximizes data usage and improves rare disease detection.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid, HybridLoss
from graph_learning.temporal_transformer import MaskedPretrainer
from data_integration.nhanes_csv_loader import NHANESCSVLoader
from data_integration.feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHANESPretrainingDataset(Dataset):
    """
    Dataset for pretraining on ALL NHANES patients (135K)
    
    Allows missing features - masked pretraining handles this
    """
    
    def __init__(self, data_path: str, seq_len: int = 12):
        self.seq_len = seq_len
        
        # Load processed patients from pickle file
        logger.info(f"Loading pretraining data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.patients = data['patients']
        logger.info(f"Loaded {len(self.patients)} patients for pretraining")
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict:
        patient = self.patients[idx]
        
        # Use pre-extracted graph features from processed data
        graph_features = patient['graph_features']
        
        # Create temporal sequence (baseline + small variations)
        organ_sequence = {}
        for organ, features in graph_features.items():
            seq = []
            for t in range(self.seq_len):
                noise = np.random.randn(len(features)) * 0.05 * (t / self.seq_len)
                seq.append(features + noise)
            organ_sequence[organ] = np.stack(seq)
        
        # Time deltas
        time_deltas = np.linspace(0, 60, self.seq_len).astype(np.float32)
        
        return {
            'organ_features': organ_sequence,
            'time_deltas': time_deltas,
            'patient_id': patient.get('patient_id', f'P{idx:06d}')
        }


class WeightedDiseaseDataset(Dataset):
    """
    Dataset for supervised training with rare disease oversampling
    """
    
    def __init__(self, data_path: str, seq_len: int = 12):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.patients = data['patients']
        self.seq_len = seq_len
        self.feature_extractor = FeatureExtractor()
        
        # Calculate disease prevalence for weighting
        self.disease_prevalence = self._calculate_prevalence()
        self.sample_weights = self._calculate_sample_weights()
        
        logger.info(f"Loaded {len(self.patients)} patients for supervised training")
        logger.info(f"Rare diseases (<5%): {sum(1 for p in self.disease_prevalence.values() if p < 0.05)}")
    
    def _calculate_prevalence(self) -> Dict[str, float]:
        """Calculate disease prevalence"""
        disease_counts = Counter()
        
        for patient in self.patients:
            for disease, has_disease in patient['disease_labels'].items():
                if has_disease:
                    disease_counts[disease] += 1
        
        prevalence = {
            disease: count / len(self.patients)
            for disease, count in disease_counts.items()
        }
        
        return prevalence
    
    def _calculate_sample_weights(self) -> np.ndarray:
        """
        Calculate sample weights for oversampling rare diseases
        
        Patients with rare diseases get higher weight
        """
        weights = []
        
        for patient in self.patients:
            # Base weight
            weight = 1.0
            
            # Increase weight for rare diseases
            for disease, has_disease in patient['disease_labels'].items():
                if has_disease:
                    prevalence = self.disease_prevalence.get(disease, 0.1)
                    if prevalence < 0.05:  # Rare disease
                        # Inverse prevalence weighting
                        weight += (0.05 / max(prevalence, 0.001))
            
            weights.append(weight)
        
        return np.array(weights)
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler for DataLoader"""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict:
        patient = self.patients[idx]
        
        # Create temporal sequence
        organ_sequence = {}
        for organ, features in patient['graph_features'].items():
            seq = []
            for t in range(self.seq_len):
                noise = np.random.randn(len(features)) * 0.05 * (t / self.seq_len)
                seq.append(features + noise)
            organ_sequence[organ] = np.stack(seq)
        
        time_deltas = np.linspace(0, 60, self.seq_len).astype(np.float32)
        
        # Demographics
        demographics = np.array([
            patient.get('age', 50) / 100,
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
        
        # Disease labels
        disease_labels = np.array([
            float(patient['disease_labels'].get(disease, False))
            for disease in GNNTransformerHybrid.DISEASE_NAMES
        ], dtype=np.float32)
        
        # Time to onset
        time_to_onset = np.where(
            disease_labels > 0.5,
            0.0,
            np.random.uniform(12, 60, size=len(disease_labels))
        ).astype(np.float32)
        
        return {
            'organ_features': organ_sequence,
            'time_deltas': time_deltas,
            'demographics': demographics,
            'disease_labels': disease_labels,
            'time_to_onset': time_to_onset,
            'patient_id': patient.get('patient_id')
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function"""
    organ_names = list(batch[0]['organ_features'].keys())
    
    organ_features = {
        organ: torch.FloatTensor(np.stack([s['organ_features'][organ] for s in batch]))
        for organ in organ_names
    }
    
    time_deltas = torch.FloatTensor(np.stack([s['time_deltas'] for s in batch]))
    
    result = {
        'organ_features': organ_features,
        'time_deltas': time_deltas,
        'patient_ids': [s['patient_id'] for s in batch]
    }
    
    # Add supervised labels if present
    if 'demographics' in batch[0]:
        result['demographics'] = torch.FloatTensor(np.stack([s['demographics'] for s in batch]))
        result['disease_labels'] = torch.FloatTensor(np.stack([s['disease_labels'] for s in batch]))
        result['time_to_onset'] = torch.FloatTensor(np.stack([s['time_to_onset'] for s in batch]))
    
    return result


def stage1_pretraining(
    model: GNNTransformerHybrid,
    data_path: str,
    max_epochs: int = 30,
    early_stop_patience: int = 5,
    batch_size: int = 128,
    device: str = 'cuda',
    save_dir: str = './models/pretrained'
):
    """
    Stage 1: Self-supervised pretraining on FULL 135K dataset
    """
    logger.info("="*80)
    logger.info("STAGE 1: SELF-SUPERVISED PRETRAINING (135K patients)")
    logger.info("="*80)
    
    # Create dataset (loads ALL patients)
    dataset = NHANESPretrainingDataset(data_path, seq_len=12)
    
    # Split for validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create pretrainer
    pretrainer = MaskedPretrainer(
        transformer=model.transformer,
        organ_embedding_dim=model.gnn.hidden_dim,
        num_organs=model.num_organs,
        mask_prob=0.15
    ).to(device)
    
    optimizer = optim.AdamW(pretrainer.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training with early stopping: max_epochs={max_epochs}, patience={early_stop_patience}")
    
    for epoch in range(max_epochs):
        # Training
        pretrainer.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{max_epochs}"):
            organ_features = {k: v.to(device) for k, v in batch['organ_features'].items()}
            time_deltas = batch['time_deltas'].to(device)
            
            # Get organ embeddings from GNN
            batch_size, seq_len = time_deltas.shape
            organ_embeddings_list = []
            
            for t in range(seq_len):
                timestep_features = {
                    organ: features[:, t, :] 
                    for organ, features in organ_features.items()
                }
                # Get GNN embeddings without output projection
                x_list = []
                for organ in sorted(model.node_feature_dims.keys(), key=lambda k: model.gnn.ORGAN_INDICES[k]):
                    if organ in timestep_features:
                        x = model.gnn.input_projections[organ](timestep_features[organ])
                        x_list.append(x)
                
                x = torch.cat(x_list, dim=0)
                edge_index = model.edge_index
                
                # Apply GNN layers
                for i, (gat, norm) in enumerate(zip(model.gnn.gat_layers, model.gnn.layer_norms)):
                    x_new = gat(x, edge_index)
                    if i > 0 and x_new.shape == x.shape:
                        x_new = x_new + x
                    x_new = norm(x_new)
                    x_new = torch.nn.functional.elu(x_new)
                    x_new = model.gnn.dropout(x_new)
                    x = x_new
                
                # Split back into organs and stack
                start_idx = 0
                organ_emb_list = []
                for organ in sorted(model.node_feature_dims.keys(), key=lambda k: model.gnn.ORGAN_INDICES[k]):
                    if organ in timestep_features:
                        num_nodes = timestep_features[organ].shape[0]
                        end_idx = start_idx + num_nodes
                        organ_emb_list.append(x[start_idx:end_idx])
                        start_idx = end_idx
                
                organ_emb = torch.stack(organ_emb_list, dim=1)  # [batch, num_organs, hidden_dim]
                organ_embeddings_list.append(organ_emb)
            
            # Stack along time dimension: [batch, seq_len, num_organs, hidden_dim]
            organ_embeddings = torch.stack(organ_embeddings_list, dim=1)
            
            # Masked pretraining
            loss, _, _ = pretrainer(organ_embeddings, time_deltas)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        pretrainer.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                organ_features = {k: v.to(device) for k, v in batch['organ_features'].items()}
                time_deltas = batch['time_deltas'].to(device)
                
                batch_size, seq_len = time_deltas.shape
                organ_embeddings_list = []
                
                for t in range(seq_len):
                    timestep_features = {
                        organ: features[:, t, :] 
                        for organ, features in organ_features.items()
                    }
                    # Get GNN embeddings without output projection
                    x_list = []
                    for organ in sorted(model.node_feature_dims.keys(), key=lambda k: model.gnn.ORGAN_INDICES[k]):
                        if organ in timestep_features:
                            x = model.gnn.input_projections[organ](timestep_features[organ])
                            x_list.append(x)
                    
                    x = torch.cat(x_list, dim=0)
                    edge_index = model.edge_index
                    
                    # Apply GNN layers
                    for i, (gat, norm) in enumerate(zip(model.gnn.gat_layers, model.gnn.layer_norms)):
                        x_new = gat(x, edge_index)
                        if i > 0 and x_new.shape == x.shape:
                            x_new = x_new + x
                        x_new = norm(x_new)
                        x_new = torch.nn.functional.elu(x_new)
                        x_new = model.gnn.dropout(x_new)
                        x = x_new
                    
                    # Split back into organs and stack
                    start_idx = 0
                    organ_emb_list = []
                    for organ in sorted(model.node_feature_dims.keys(), key=lambda k: model.gnn.ORGAN_INDICES[k]):
                        if organ in timestep_features:
                            num_nodes = timestep_features[organ].shape[0]
                            end_idx = start_idx + num_nodes
                            organ_emb_list.append(x[start_idx:end_idx])
                            start_idx = end_idx
                    
                    organ_emb = torch.stack(organ_emb_list, dim=1)
                    organ_embeddings_list.append(organ_emb)
                
                organ_embeddings = torch.stack(organ_embeddings_list, dim=1)
                loss, _, _ = pretrainer(organ_embeddings, time_deltas)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{max_epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'pretrainer_state_dict': pretrainer.state_dict(),
                'gnn_state_dict': model.gnn.state_dict(),  # Save GNN weights!
                'transformer_state_dict': model.transformer.state_dict(),  # Save transformer separately too
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path / 'pretrained_best.pt')
            logger.info(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            logger.info(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            logger.info(f"   Best val loss: {best_val_loss:.4f}")
            break
        
        scheduler.step()
        
    logger.info("Stage 1 complete!")
    return model


def stage2_finetuning(
    model: GNNTransformerHybrid,
    data_path: str,
    max_epochs: int = 150,
    early_stop_patience: int = 15,
    batch_size: int = 64,
    device: str = 'cuda',
    save_dir: str = './models/finetuned'
):
    """
    Stage 2: Supervised fine-tuning on 67K complete dataset with rare disease handling
    """
    logger.info("="*80)
    logger.info("STAGE 2: SUPERVISED FINE-TUNING (67K complete patients)")
    logger.info("="*80)
    
    # Create dataset with rare disease weighting
    dataset = WeightedDiseaseDataset(data_path, seq_len=12)
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create weighted sampler for training subset only
    train_indices = train_dataset.indices
    train_weights = dataset.sample_weights[train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Calculate class weights for rare diseases
    disease_weights = torch.FloatTensor([
        1.0 / max(dataset.disease_prevalence.get(disease, 0.1), 0.01)
        for disease in GNNTransformerHybrid.DISEASE_NAMES
    ]).to(device)
    
    # Normalize weights
    disease_weights = disease_weights / disease_weights.mean()
    
    logger.info(f"Disease weights (top 5 rare): {disease_weights.topk(5).values}")
    
    # Weighted loss
    criterion = HybridLoss(disease_weights=disease_weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_val_auc = 0.0
    epochs_without_improvement = 0
    
    logger.info(f"Training with early stopping: max_epochs={max_epochs}, patience={early_stop_patience}")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{max_epochs}"):
            organ_features = {k: v.to(device) for k, v in batch['organ_features'].items()}
            time_deltas = batch['time_deltas'].to(device)
            demographics = batch['demographics'].to(device)
            disease_labels = batch['disease_labels'].to(device)
            time_to_onset = batch['time_to_onset'].to(device)
            
            predictions, _ = model(organ_features, time_deltas, demographics)
            
            targets = {
                'disease_labels': disease_labels,
                'time_to_onset': time_to_onset
            }
            
            loss, _ = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                organ_features = {k: v.to(device) for k, v in batch['organ_features'].items()}
                time_deltas = batch['time_deltas'].to(device)
                demographics = batch['demographics'].to(device)
                disease_labels = batch['disease_labels'].to(device)
                time_to_onset = batch['time_to_onset'].to(device)
                
                predictions, _ = model(organ_features, time_deltas, demographics)
                
                targets = {
                    'disease_labels': disease_labels,
                    'time_to_onset': time_to_onset
                }
                
                loss, _ = criterion(predictions, targets)
                val_loss += loss.item()
                
                all_preds.append(predictions['risk_scores'].cpu().numpy())
                all_labels.append(disease_labels.cpu().numpy())
        
        # Calculate AUC
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        disease_aucs = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                disease_aucs.append(auc)
        
        mean_auc = np.mean(disease_aucs)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{max_epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Mean AUC: {mean_auc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': mean_auc
            }, save_path / 'best_model.pt')
            logger.info(f"  ✓ Saved best model (AUC: {mean_auc:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            logger.info(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            logger.info(f"   Best val AUC: {best_val_auc:.4f}")
            break
    
    logger.info("Stage 2 complete!")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_data', default='./data/nhanes_all_135310.pkl')
    parser.add_argument('--finetune_data', default='./data/nhanes_all_135310.pkl')
    parser.add_argument('--max_epochs', type=int, default=100)
    args = parser.parse_args()
    
    config = {
        'nhanes_pretrain_path': args.pretrain_data,  # Use hybrid dataset
        'nhanes_finetune_path': args.finetune_data,  # Use hybrid temporal dataset
        'pretrain_max_epochs': args.max_epochs,
        'pretrain_patience': 10,
        'finetune_max_epochs': 150,
        'finetune_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("="*80)
    logger.info("TWO-STAGE TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Device: {config['device']}")
    
    # Create model
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
    
    # Stage 2: Fine-tune on 135K patients (with disease labels)
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
    
    logger.info("\n✓ Two-stage training complete!")


if __name__ == '__main__':
    main()
