#!/usr/bin/env python3
"""
Example Patient Disease Prediction Testing

Load trained model and predict disease risks for example patients with
different health profiles to demonstrate the model's capabilities.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid


class PatientPredictor:
    """Generate disease predictions for example patients"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Disease names (24 diseases)
        self.disease_names = [
            'Diabetes', 'Hypertension', 'Heart_Disease', 'Stroke',
            'Kidney_Disease', 'Liver_Disease', 'COPD', 'Asthma',
            'Cancer', 'Arthritis', 'Osteoporosis', 'Depression',
            'Anxiety', 'Dementia', 'Obesity', 'Metabolic_Syndrome',
            'Thyroid_Disease', 'Anemia', 'Sleep_Apnea', 'Gout',
            'Hepatitis', 'Cirrhosis', 'Heart_Failure', 'Atrial_Fibrillation'
        ]
        
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
    
    def predict_patient(self, organ_features, patient_name: str):
        """Predict disease risks for a patient"""
        print(f"\n{'='*80}")
        print(f"PATIENT: {patient_name}")
        print(f"{'='*80}")
        
        # Display organ states
        print(f"\nOrgan States:")
        print(f"  Metabolic:      Glucose={organ_features['metabolic'][0, 0]:.1f} mg/dL, HbA1c={organ_features['metabolic'][0, 1]:.1f}%")
        print(f"  Cardiovascular: BP={organ_features['cardiovascular'][0, 0]:.0f}/{organ_features['cardiovascular'][0, 1]:.0f} mmHg, Chol={organ_features['cardiovascular'][0, 2]:.0f} mg/dL")
        print(f"  Kidney:         Creatinine={organ_features['kidney'][0, 0]:.2f} mg/dL")
        print(f"  Liver:          ALT={organ_features['liver'][0, 0]:.1f} U/L, AST={organ_features['liver'][0, 1]:.1f} U/L")
        print(f"  Immune:         WBC={organ_features['immune'][0, 0]:.1f} K/μL")
        print(f"  Neural:         Cognitive={organ_features['neural'][0, 0]:.2f}")
        print(f"  Lifestyle:      Alcohol={organ_features['lifestyle'][0, 0]:.2f}, Exercise={organ_features['lifestyle'][0, 1]:.2f}")
        
        # Predict
        with torch.no_grad():
            # Create temporal sequence (single timepoint for now)
            temporal_features = {}
            for organ, features in organ_features.items():
                temporal_features[organ] = features.unsqueeze(1)  # [batch, seq_len=1, features]
            
            # Forward pass
            outputs = self.model(temporal_features, self.edge_index)
            
            # Get disease predictions
            disease_risks = torch.sigmoid(outputs['disease_risk']).cpu().numpy()[0]
        
        # Display top 10 disease risks
        print(f"\nTop 10 Disease Risks:")
        risk_indices = np.argsort(disease_risks)[::-1][:10]
        
        for i, idx in enumerate(risk_indices, 1):
            risk = disease_risks[idx]
            disease = self.disease_names[idx]
            risk_level = self.get_risk_level(risk)
            print(f"  {i}. {disease:25s} {risk:.1%}  [{risk_level}]")
        
        return disease_risks
    
    def get_risk_level(self, risk: float) -> str:
        """Get risk level label"""
        if risk >= 0.7:
            return "HIGH RISK"
        elif risk >= 0.4:
            return "MODERATE"
        elif risk >= 0.2:
            return "LOW-MOD"
        else:
            return "LOW"
    
    def create_example_patients(self):
        """Create example patients with different profiles"""
        
        # Patient 1: Healthy Young Adult
        patient1 = {
            'metabolic': torch.tensor([[95.0, 5.2, 12.0, 120.0]], dtype=torch.float32),
            'cardiovascular': torch.tensor([[115.0, 75.0, 170.0, 55.0, 100.0]], dtype=torch.float32),
            'kidney': torch.tensor([[0.9, 12.0]], dtype=torch.float32),
            'liver': torch.tensor([[22.0, 18.0]], dtype=torch.float32),
            'immune': torch.tensor([[6.5]], dtype=torch.float32),
            'neural': torch.tensor([[0.95]], dtype=torch.float32),
            'lifestyle': torch.tensor([[0.2, 0.7, 0.7, 7.5]], dtype=torch.float32)
        }
        
        # Patient 2: Pre-Diabetic with Metabolic Syndrome
        patient2 = {
            'metabolic': torch.tensor([[125.0, 6.2, 25.0, 220.0]], dtype=torch.float32),
            'cardiovascular': torch.tensor([[135.0, 88.0, 240.0, 38.0, 160.0]], dtype=torch.float32),
            'kidney': torch.tensor([[1.1, 18.0]], dtype=torch.float32),
            'liver': torch.tensor([[45.0, 38.0]], dtype=torch.float32),
            'immune': torch.tensor([[8.5]], dtype=torch.float32),
            'neural': torch.tensor([[0.85]], dtype=torch.float32),
            'lifestyle': torch.tensor([[0.4, 0.2, 0.3, 6.0]], dtype=torch.float32)
        }
        
        # Patient 3: Heavy Drinker with Liver Issues
        patient3 = {
            'metabolic': torch.tensor([[105.0, 5.6, 18.0, 180.0]], dtype=torch.float32),
            'cardiovascular': torch.tensor([[128.0, 82.0, 195.0, 48.0, 125.0]], dtype=torch.float32),
            'kidney': torch.tensor([[1.0, 15.0]], dtype=torch.float32),
            'liver': torch.tensor([[68.0, 55.0]], dtype=torch.float32),
            'immune': torch.tensor([[9.2]], dtype=torch.float32),
            'neural': torch.tensor([[0.78]], dtype=torch.float32),
            'lifestyle': torch.tensor([[0.9, 0.3, 0.4, 6.5]], dtype=torch.float32)
        }
        
        # Patient 4: Elderly with Hypertension and Cognitive Decline
        patient4 = {
            'metabolic': torch.tensor([[110.0, 5.8, 16.0, 165.0]], dtype=torch.float32),
            'cardiovascular': torch.tensor([[155.0, 92.0, 210.0, 42.0, 145.0]], dtype=torch.float32),
            'kidney': torch.tensor([[1.3, 22.0]], dtype=torch.float32),
            'liver': torch.tensor([[28.0, 24.0]], dtype=torch.float32),
            'immune': torch.tensor([[7.8]], dtype=torch.float32),
            'neural': torch.tensor([[0.65]], dtype=torch.float32),
            'lifestyle': torch.tensor([[0.1, 0.3, 0.5, 7.0]], dtype=torch.float32)
        }
        
        # Patient 5: Athletic with Excellent Health
        patient5 = {
            'metabolic': torch.tensor([[88.0, 5.0, 10.0, 95.0]], dtype=torch.float32),
            'cardiovascular': torch.tensor([[110.0, 70.0, 155.0, 62.0, 85.0]], dtype=torch.float32),
            'kidney': torch.tensor([[0.85, 11.0]], dtype=torch.float32),
            'liver': torch.tensor([[18.0, 16.0]], dtype=torch.float32),
            'immune': torch.tensor([[6.0]], dtype=torch.float32),
            'neural': torch.tensor([[0.98]], dtype=torch.float32),
            'lifestyle': torch.tensor([[0.1, 0.9, 0.8, 8.0]], dtype=torch.float32)
        }
        
        # Move to device
        for patient in [patient1, patient2, patient3, patient4, patient5]:
            for organ in patient:
                patient[organ] = patient[organ].to(self.device)
        
        return [
            (patient1, "Healthy Young Adult (Age 28)"),
            (patient2, "Pre-Diabetic with Metabolic Syndrome (Age 52)"),
            (patient3, "Heavy Drinker with Liver Issues (Age 45)"),
            (patient4, "Elderly with Hypertension & Cognitive Decline (Age 72)"),
            (patient5, "Athletic with Excellent Health (Age 35)")
        ]
    
    def run_predictions(self):
        """Run predictions for all example patients"""
        print("\n" + "="*80)
        print("EXAMPLE PATIENT DISEASE PREDICTIONS")
        print("="*80)
        print("\nDemonstrating multi-organ disease risk prediction...")
        
        patients = self.create_example_patients()
        all_results = []
        
        for organ_features, patient_name in patients:
            risks = self.predict_patient(organ_features, patient_name)
            all_results.append({
                'name': patient_name,
                'risks': risks
            })
        
        # Summary comparison
        print(f"\n{'='*80}")
        print("COMPARATIVE RISK SUMMARY")
        print(f"{'='*80}")
        
        # Create comparison table for top diseases
        top_diseases = ['Diabetes', 'Hypertension', 'Heart_Disease', 'Liver_Disease', 'Dementia']
        
        print(f"\n{'Patient':<50s} " + " ".join([f"{d:>12s}" for d in top_diseases]))
        print("-" * 120)
        
        for result in all_results:
            name = result['name']
            risks = result['risks']
            
            risk_str = name[:48] + "  "
            for disease in top_diseases:
                idx = self.disease_names.index(disease)
                risk_str += f"{risks[idx]:>12.1%} "
            
            print(risk_str)
        
        return all_results


def main():
    """Run example patient predictions"""
    
    model_path = './models/finetuned/best_model.pt'
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    predictor = PatientPredictor(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results = predictor.run_predictions()
    
    # Save results
    output_path = './validation/example_patient_predictions.pkl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
