#!/usr/bin/env python3
"""
Cross-Organ Coupling Validation via Perturbation Tests

Tests whether the trained GNN-Transformer model has learned physiological
cross-organ interactions by perturbing one organ and measuring effects on others.

Expected physiological responses:
1. Glucose ↑ → ALT ↑ (metabolic-liver coupling)
2. BP ↑ → Cognitive ↓ (cardiovascular-neural coupling)
3. Exercise ↑ → Multi-organ benefits (glucose↓, ALT↓, WBC↓, cognitive↑)
4. Alcohol ↑ → ALT ↑ (lifestyle-liver coupling)
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_learning.gnn_transformer_hybrid import GNNTransformerHybrid


class CrossOrganCouplingValidator:
    """Validate learned cross-organ interactions via perturbation tests"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Organ feature dimensions
        self.node_dims = {
            'metabolic': 4,      # glucose, HbA1c, insulin, triglycerides
            'cardiovascular': 5, # systolic_bp, diastolic_bp, total_chol, HDL, LDL
            'kidney': 2,         # creatinine, BUN
            'liver': 2,          # ALT, AST
            'immune': 1,         # WBC
            'neural': 1,         # cognitive_score
            'lifestyle': 4       # alcohol, exercise, diet, sleep
        }
        
        # Edge index for organ graph
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
        
        print(f"  ✓ Model loaded successfully")
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
            edge_list.append([organ_to_idx[dst], organ_to_idx[src]])  # Bidirectional
        
        return torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
    
    def create_baseline_patient(self):
        """Create a baseline healthy patient"""
        baseline = {
            'metabolic': torch.tensor([100.0, 5.5, 15.0, 150.0], dtype=torch.float32),  # Normal glucose, HbA1c, insulin, triglycerides
            'cardiovascular': torch.tensor([120.0, 80.0, 180.0, 50.0, 110.0], dtype=torch.float32),  # Normal BP, cholesterol
            'kidney': torch.tensor([1.0, 15.0], dtype=torch.float32),  # Normal creatinine, BUN
            'liver': torch.tensor([25.0, 20.0], dtype=torch.float32),  # Normal ALT, AST
            'immune': torch.tensor([7.0], dtype=torch.float32),  # Normal WBC
            'neural': torch.tensor([0.90], dtype=torch.float32),  # Good cognitive function
            'lifestyle': torch.tensor([0.3, 0.5, 0.6, 7.0], dtype=torch.float32)  # Moderate alcohol, exercise, diet, sleep
        }
        
        # Add batch dimension and move to device
        for organ in baseline:
            baseline[organ] = baseline[organ].unsqueeze(0).to(self.device)
        
        return baseline
    
    def predict_organs(self, organ_features):
        """Predict organ states using the model"""
        with torch.no_grad():
            # Forward pass through GNN
            gnn_outputs = self.model.gnn(organ_features, self.edge_index)
            
            # Extract organ predictions (use GNN output projections)
            predictions = {}
            for organ, features in gnn_outputs.items():
                predictions[organ] = features.cpu().numpy()[0]
        
        return predictions
    
    def perturbation_test_glucose_alt(self):
        """
        Test 1: Glucose → ALT coupling
        
        Hypothesis: Increasing glucose should increase ALT (metabolic-liver coupling)
        """
        print("\n" + "="*80)
        print("TEST 1: Glucose → ALT Coupling (Metabolic-Liver)")
        print("="*80)
        
        baseline = self.create_baseline_patient()
        baseline_pred = self.predict_organs(baseline)
        
        # Perturbation: Increase glucose by 50 mg/dL
        perturbed = {k: v.clone() for k, v in baseline.items()}
        perturbed['metabolic'][0, 0] += 50.0  # Glucose: 100 → 150 mg/dL
        
        perturbed_pred = self.predict_organs(perturbed)
        
        # Measure changes
        baseline_glucose = baseline['metabolic'][0, 0].item()
        perturbed_glucose = perturbed['metabolic'][0, 0].item()
        baseline_alt = baseline_pred['liver'][0]
        perturbed_alt = perturbed_pred['liver'][0]
        
        delta_glucose = perturbed_glucose - baseline_glucose
        delta_alt = perturbed_alt - baseline_alt
        
        print(f"\nPerturbation:")
        print(f"  Glucose: {baseline_glucose:.1f} → {perturbed_glucose:.1f} mg/dL (Δ = +{delta_glucose:.1f})")
        
        print(f"\nObserved Response:")
        print(f"  ALT: {baseline_alt:.1f} → {perturbed_alt:.1f} U/L (Δ = {delta_alt:+.1f})")
        
        # Validation
        expected_direction = "increase"
        observed_direction = "increase" if delta_alt > 0 else "decrease"
        
        print(f"\nValidation:")
        print(f"  Expected: ALT should {expected_direction}")
        print(f"  Observed: ALT {observed_direction}d")
        
        if delta_alt > 0:
            print(f"  ✓ PASS: Model learned metabolic-liver coupling")
            print(f"  Effect size: {delta_alt/delta_glucose:.3f} U/L per mg/dL glucose")
        else:
            print(f"  ✗ FAIL: Model did not learn expected coupling")
        
        return {
            'test': 'glucose_alt',
            'delta_input': delta_glucose,
            'delta_output': delta_alt,
            'passed': delta_alt > 0
        }
    
    def perturbation_test_bp_cognitive(self):
        """
        Test 2: BP → Cognitive coupling
        
        Hypothesis: Increasing BP should decrease cognitive score (vascular damage)
        """
        print("\n" + "="*80)
        print("TEST 2: BP → Cognitive Coupling (Cardiovascular-Neural)")
        print("="*80)
        
        baseline = self.create_baseline_patient()
        baseline_pred = self.predict_organs(baseline)
        
        # Perturbation: Increase systolic BP by 20 mmHg
        perturbed = {k: v.clone() for k, v in baseline.items()}
        perturbed['cardiovascular'][0, 0] += 20.0  # Systolic BP: 120 → 140 mmHg
        
        perturbed_pred = self.predict_organs(perturbed)
        
        # Measure changes
        baseline_bp = baseline['cardiovascular'][0, 0].item()
        perturbed_bp = perturbed['cardiovascular'][0, 0].item()
        baseline_cog = baseline_pred['neural'][0]
        perturbed_cog = perturbed_pred['neural'][0]
        
        delta_bp = perturbed_bp - baseline_bp
        delta_cog = perturbed_cog - baseline_cog
        
        print(f"\nPerturbation:")
        print(f"  Systolic BP: {baseline_bp:.1f} → {perturbed_bp:.1f} mmHg (Δ = +{delta_bp:.1f})")
        
        print(f"\nObserved Response:")
        print(f"  Cognitive: {baseline_cog:.3f} → {perturbed_cog:.3f} (Δ = {delta_cog:+.3f})")
        
        # Validation
        expected_direction = "decrease"
        observed_direction = "decrease" if delta_cog < 0 else "increase"
        
        print(f"\nValidation:")
        print(f"  Expected: Cognitive should {expected_direction}")
        print(f"  Observed: Cognitive {observed_direction}d")
        
        if delta_cog < 0:
            print(f"  ✓ PASS: Model learned cardiovascular-neural coupling")
            print(f"  Effect size: {abs(delta_cog)/delta_bp:.4f} per mmHg")
        else:
            print(f"  ✗ FAIL: Model did not learn expected coupling")
        
        return {
            'test': 'bp_cognitive',
            'delta_input': delta_bp,
            'delta_output': delta_cog,
            'passed': delta_cog < 0
        }
    
    def perturbation_test_exercise_multi_organ(self):
        """
        Test 3: Exercise → Multi-organ effects
        
        Hypothesis: Increasing exercise should:
        - Decrease glucose (metabolic benefit)
        - Decrease ALT (liver benefit)
        - Decrease WBC (anti-inflammatory)
        - Increase cognitive (neuroprotection)
        """
        print("\n" + "="*80)
        print("TEST 3: Exercise → Multi-Organ Effects (Lifestyle-All)")
        print("="*80)
        
        baseline = self.create_baseline_patient()
        baseline_pred = self.predict_organs(baseline)
        
        # Perturbation: Increase exercise from 0.5 to 0.8
        perturbed = {k: v.clone() for k, v in baseline.items()}
        perturbed['lifestyle'][0, 1] += 0.3  # Exercise: 0.5 → 0.8
        
        perturbed_pred = self.predict_organs(perturbed)
        
        # Measure changes
        baseline_exercise = baseline['lifestyle'][0, 1].item()
        perturbed_exercise = perturbed['lifestyle'][0, 1].item()
        
        delta_exercise = perturbed_exercise - baseline_exercise
        
        # Multi-organ responses
        delta_glucose = perturbed_pred['metabolic'][0] - baseline_pred['metabolic'][0]
        delta_alt = perturbed_pred['liver'][0] - baseline_pred['liver'][0]
        delta_wbc = perturbed_pred['immune'][0] - baseline_pred['immune'][0]
        delta_cog = perturbed_pred['neural'][0] - baseline_pred['neural'][0]
        
        print(f"\nPerturbation:")
        print(f"  Exercise: {baseline_exercise:.2f} → {perturbed_exercise:.2f} (Δ = +{delta_exercise:.2f})")
        
        print(f"\nObserved Multi-Organ Responses:")
        print(f"  Glucose:   {baseline_pred['metabolic'][0]:.1f} → {perturbed_pred['metabolic'][0]:.1f} mg/dL (Δ = {delta_glucose:+.1f})")
        print(f"  ALT:       {baseline_pred['liver'][0]:.1f} → {perturbed_pred['liver'][0]:.1f} U/L (Δ = {delta_alt:+.1f})")
        print(f"  WBC:       {baseline_pred['immune'][0]:.2f} → {perturbed_pred['immune'][0]:.2f} K/μL (Δ = {delta_wbc:+.2f})")
        print(f"  Cognitive: {baseline_pred['neural'][0]:.3f} → {perturbed_pred['neural'][0]:.3f} (Δ = {delta_cog:+.3f})")
        
        # Validation
        tests = {
            'Glucose ↓': delta_glucose < 0,
            'ALT ↓': delta_alt < 0,
            'WBC ↓': delta_wbc < 0,
            'Cognitive ↑': delta_cog > 0
        }
        
        print(f"\nValidation:")
        for test_name, passed in tests.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {test_name}")
        
        passed_count = sum(tests.values())
        total_count = len(tests)
        
        print(f"\nOverall: {passed_count}/{total_count} expected effects observed")
        
        return {
            'test': 'exercise_multi_organ',
            'delta_input': delta_exercise,
            'delta_outputs': {
                'glucose': delta_glucose,
                'alt': delta_alt,
                'wbc': delta_wbc,
                'cognitive': delta_cog
            },
            'passed': passed_count >= 3  # At least 3/4 effects
        }
    
    def perturbation_test_alcohol_liver(self):
        """
        Test 4: Alcohol → ALT coupling
        
        Hypothesis: Increasing alcohol should increase ALT (hepatotoxicity)
        """
        print("\n" + "="*80)
        print("TEST 4: Alcohol → ALT Coupling (Lifestyle-Liver)")
        print("="*80)
        
        baseline = self.create_baseline_patient()
        baseline_pred = self.predict_organs(baseline)
        
        # Perturbation: Increase alcohol from 0.3 to 0.8 (heavy drinking)
        perturbed = {k: v.clone() for k, v in baseline.items()}
        perturbed['lifestyle'][0, 0] += 0.5  # Alcohol: 0.3 → 0.8
        
        perturbed_pred = self.predict_organs(perturbed)
        
        # Measure changes
        baseline_alcohol = baseline['lifestyle'][0, 0].item()
        perturbed_alcohol = perturbed['lifestyle'][0, 0].item()
        baseline_alt = baseline_pred['liver'][0]
        perturbed_alt = perturbed_pred['liver'][0]
        
        delta_alcohol = perturbed_alcohol - baseline_alcohol
        delta_alt = perturbed_alt - baseline_alt
        
        print(f"\nPerturbation:")
        print(f"  Alcohol: {baseline_alcohol:.2f} → {perturbed_alcohol:.2f} (Δ = +{delta_alcohol:.2f})")
        
        print(f"\nObserved Response:")
        print(f"  ALT: {baseline_alt:.1f} → {perturbed_alt:.1f} U/L (Δ = {delta_alt:+.1f})")
        
        # Validation
        expected_direction = "increase"
        observed_direction = "increase" if delta_alt > 0 else "decrease"
        
        print(f"\nValidation:")
        print(f"  Expected: ALT should {expected_direction}")
        print(f"  Observed: ALT {observed_direction}d")
        
        if delta_alt > 0:
            print(f"  ✓ PASS: Model learned lifestyle-liver coupling")
            print(f"  Effect size: {delta_alt/delta_alcohol:.1f} U/L per unit alcohol")
        else:
            print(f"  ✗ FAIL: Model did not learn expected coupling")
        
        return {
            'test': 'alcohol_alt',
            'delta_input': delta_alcohol,
            'delta_output': delta_alt,
            'passed': delta_alt > 0
        }
    
    def run_all_tests(self):
        """Run all perturbation tests"""
        print("\n" + "="*80)
        print("CROSS-ORGAN COUPLING VALIDATION")
        print("="*80)
        print("\nTesting whether GNN learned physiological organ interactions...")
        
        results = []
        
        # Run tests
        results.append(self.perturbation_test_glucose_alt())
        results.append(self.perturbation_test_bp_cognitive())
        results.append(self.perturbation_test_exercise_multi_organ())
        results.append(self.perturbation_test_alcohol_liver())
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        
        print(f"\nTests Passed: {passed}/{total}")
        
        for result in results:
            status = "✓" if result['passed'] else "✗"
            print(f"  {status} {result['test']}")
        
        if passed == total:
            print(f"\n✓ ALL TESTS PASSED")
            print(f"  Model successfully learned cross-organ physiological coupling!")
        elif passed >= total * 0.75:
            print(f"\n⚠ MOSTLY PASSED ({passed}/{total})")
            print(f"  Model learned most cross-organ interactions")
        else:
            print(f"\n✗ TESTS FAILED ({passed}/{total})")
            print(f"  Model may not have learned expected coupling")
        
        return results


def main():
    """Run cross-organ coupling validation"""
    
    # Find best model
    model_path = './models/finetuned/best_model.pt'
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please ensure Stage 2 training completed successfully")
        return
    
    # Run validation
    validator = CrossOrganCouplingValidator(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results = validator.run_all_tests()
    
    # Save results
    output_path = './validation/coupling_test_results.pkl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
