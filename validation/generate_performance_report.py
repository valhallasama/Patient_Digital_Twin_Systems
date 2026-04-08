#!/usr/bin/env python3
"""
Performance Report Generation

Generate comprehensive performance metrics for the trained model:
- Per-disease AUC scores
- Confusion matrices
- Precision, Recall, F1 scores
- ROC curves
- Calibration plots
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_fscore_support, classification_report
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid
from train_two_stage import WeightedDiseaseDataset


class PerformanceReporter:
    """Generate comprehensive performance report"""
    
    def __init__(self, model_path: str, data_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.disease_names = [
            'Diabetes', 'Hypertension', 'Heart_Disease', 'Stroke',
            'Kidney_Disease', 'Liver_Disease', 'COPD', 'Asthma',
            'Cancer', 'Arthritis', 'Osteoporosis', 'Depression',
            'Anxiety', 'Dementia', 'Obesity', 'Metabolic_Syndrome',
            'Thyroid_Disease', 'Anemia', 'Sleep_Apnea', 'Gout',
            'Hepatitis', 'Cirrhosis', 'Heart_Failure', 'Atrial_Fibrillation'
        ]
        
        self.data_path = data_path
        self.edge_index = self.create_edge_index()
    
    def load_model(self, model_path: str):
        """Load trained model"""
        print(f"Loading model from {model_path}...")
        
        node_dims = {
            'metabolic': 4,
            'cardiovascular': 5,
            'kidney': 2,
            'liver': 2,
            'immune': 1,
            'neural': 1,
            'lifestyle': 4
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
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # Handle checkpoint dictionary structure
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"  ✓ Model loaded")
        return model
    
    def create_edge_index(self):
        """Create edge index for organ graph"""
        organ_to_idx = {
            'metabolic': 0, 'cardiovascular': 1, 'kidney': 2,
            'liver': 3, 'immune': 4, 'neural': 5, 'lifestyle': 6
        }
        
        edges = [
            ('metabolic', 'cardiovascular'),
            ('metabolic', 'liver'),
            ('metabolic', 'kidney'),
            ('cardiovascular', 'kidney'),
            ('cardiovascular', 'neural'),
            ('liver', 'immune'),
            ('lifestyle', 'metabolic'),
            ('lifestyle', 'cardiovascular'),
            ('lifestyle', 'liver'),
            ('lifestyle', 'immune'),
            ('lifestyle', 'neural')
        ]
        
        edge_list = []
        for src, dst in edges:
            edge_list.append([organ_to_idx[src], organ_to_idx[dst]])
            edge_list.append([organ_to_idx[dst], organ_to_idx[src]])
        
        return torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
    
    def evaluate_on_dataset(self, split: str = 'test'):
        """Evaluate model on dataset"""
        print(f"\nEvaluating on {split} set...")
        
        # Load dataset
        dataset = WeightedDiseaseDataset(self.data_path, seq_len=12)
        
        # Use last 20% as test set
        test_size = int(0.2 * len(dataset))
        test_indices = list(range(len(dataset) - test_size, len(dataset)))
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for idx in tqdm(test_indices, desc="Evaluating"):
                sample = dataset[idx]
                
                # Prepare input
                temporal_features = {}
                for organ in ['metabolic', 'cardiovascular', 'kidney', 'liver', 'immune', 'neural', 'lifestyle']:
                    feat = sample[organ].unsqueeze(0).to(self.device)  # [1, seq_len, features]
                    temporal_features[organ] = feat
                
                labels = sample['disease_labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(temporal_features, self.edge_index)
                predictions = torch.sigmoid(outputs['disease_risk'])
                
                all_predictions.append(predictions.cpu().numpy()[0])
                all_labels.append(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        print(f"  ✓ Evaluated {len(test_indices)} samples")
        
        return all_predictions, all_labels
    
    def compute_per_disease_metrics(self, predictions, labels):
        """Compute metrics for each disease"""
        print("\nComputing per-disease metrics...")
        
        metrics = []
        
        for i, disease_name in enumerate(self.disease_names):
            y_true = labels[:, i]
            y_pred = predictions[:, i]
            
            # Skip if no positive samples
            if y_true.sum() == 0:
                metrics.append({
                    'disease': disease_name,
                    'auc': np.nan,
                    'prevalence': 0.0,
                    'n_positive': 0,
                    'n_negative': len(y_true)
                })
                continue
            
            # Compute AUC
            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = np.nan
            
            # Binary predictions (threshold 0.5)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_binary, average='binary', zero_division=0
            )
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            
            metrics.append({
                'disease': disease_name,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'prevalence': y_true.mean(),
                'n_positive': int(y_true.sum()),
                'n_negative': int((1 - y_true).sum()),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            })
        
        return metrics
    
    def plot_per_disease_auc(self, metrics, output_dir: Path):
        """Plot AUC scores for each disease"""
        print("\nGenerating AUC plot...")
        
        # Filter out diseases with NaN AUC
        valid_metrics = [m for m in metrics if not np.isnan(m['auc'])]
        
        # Sort by AUC
        valid_metrics = sorted(valid_metrics, key=lambda x: x['auc'], reverse=True)
        
        diseases = [m['disease'] for m in valid_metrics]
        aucs = [m['auc'] for m in valid_metrics]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        colors = ['green' if auc >= 0.8 else 'orange' if auc >= 0.7 else 'red' for auc in aucs]
        
        plt.barh(diseases, aucs, color=colors, alpha=0.7)
        plt.xlabel('AUC Score', fontsize=12)
        plt.ylabel('Disease', fontsize=12)
        plt.title('Per-Disease AUC Scores', fontsize=14, fontweight='bold')
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
        plt.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.7)')
        plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
        plt.xlim(0.4, 1.0)
        plt.legend()
        plt.tight_layout()
        
        output_path = output_dir / 'per_disease_auc.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")
    
    def plot_confusion_matrices(self, metrics, output_dir: Path):
        """Plot confusion matrices for top diseases"""
        print("\nGenerating confusion matrices...")
        
        # Select top 6 diseases by prevalence
        top_diseases = sorted(
            [m for m in metrics if m['n_positive'] > 0],
            key=lambda x: x['prevalence'],
            reverse=True
        )[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(top_diseases):
            cm = np.array([
                [metric['tn'], metric['fp']],
                [metric['fn'], metric['tp']]
            ])
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(f"{metric['disease']}\nAUC: {metric['auc']:.3f}", fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        output_path = output_dir / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")
    
    def generate_summary_table(self, metrics, output_dir: Path):
        """Generate summary table"""
        print("\nGenerating summary table...")
        
        # Create summary
        summary = []
        for m in metrics:
            if not np.isnan(m['auc']):
                summary.append({
                    'Disease': m['disease'],
                    'AUC': f"{m['auc']:.3f}",
                    'Precision': f"{m['precision']:.3f}",
                    'Recall': f"{m['recall']:.3f}",
                    'F1': f"{m['f1']:.3f}",
                    'Prevalence': f"{m['prevalence']:.1%}",
                    'N_Positive': m['n_positive']
                })
        
        # Sort by AUC
        summary = sorted(summary, key=lambda x: float(x['AUC']), reverse=True)
        
        # Print table
        print("\n" + "="*100)
        print("PER-DISEASE PERFORMANCE SUMMARY")
        print("="*100)
        print(f"{'Disease':<25s} {'AUC':>8s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s} {'Prevalence':>12s} {'N_Pos':>8s}")
        print("-"*100)
        
        for row in summary:
            print(f"{row['Disease']:<25s} {row['AUC']:>8s} {row['Precision']:>10s} {row['Recall']:>10s} {row['F1']:>8s} {row['Prevalence']:>12s} {row['N_Positive']:>8d}")
        
        # Overall statistics
        valid_aucs = [float(row['AUC']) for row in summary]
        print("-"*100)
        print(f"{'OVERALL MEAN':<25s} {np.mean(valid_aucs):>8.3f}")
        print(f"{'OVERALL MEDIAN':<25s} {np.median(valid_aucs):>8.3f}")
        print(f"{'DISEASES >= 0.80 AUC':<25s} {sum(1 for auc in valid_aucs if auc >= 0.80):>8d} / {len(valid_aucs)}")
        print(f"{'DISEASES >= 0.70 AUC':<25s} {sum(1 for auc in valid_aucs if auc >= 0.70):>8d} / {len(valid_aucs)}")
        print("="*100)
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(summary)
        output_path = output_dir / 'performance_summary.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Saved to {output_path}")
        
        return summary
    
    def generate_report(self):
        """Generate complete performance report"""
        print("\n" + "="*80)
        print("PERFORMANCE REPORT GENERATION")
        print("="*80)
        
        # Create output directory
        output_dir = Path('./validation/performance_report')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate model
        predictions, labels = self.evaluate_on_dataset('test')
        
        # Compute metrics
        metrics = self.compute_per_disease_metrics(predictions, labels)
        
        # Generate visualizations
        self.plot_per_disease_auc(metrics, output_dir)
        self.plot_confusion_matrices(metrics, output_dir)
        
        # Generate summary table
        summary = self.generate_summary_table(metrics, output_dir)
        
        # Save raw metrics
        with open(output_dir / 'raw_metrics.pkl', 'wb') as f:
            pickle.dump({
                'metrics': metrics,
                'predictions': predictions,
                'labels': labels
            }, f)
        
        print(f"\n✓ Complete report saved to {output_dir}")
        
        return metrics


def main():
    """Generate performance report"""
    
    model_path = './models/finetuned/best_model.pt'
    data_path = './data/nhanes_augmented_complete.pkl'
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if not Path(data_path).exists():
        print(f"Error: Data not found at {data_path}")
        return
    
    reporter = PerformanceReporter(
        model_path=model_path,
        data_path=data_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    metrics = reporter.generate_report()


if __name__ == '__main__':
    main()
