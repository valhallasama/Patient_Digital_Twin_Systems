#!/usr/bin/env python3
"""
Improved Synthetic Data Generation with Strong Cross-Organ Coupling

Key improvements:
1. Multivariate conditional sampling (not independent generation)
2. Stronger effect sizes for cross-organ dependencies
3. AR(1) temporal dynamics with physiological constraints
4. Explicit modeling of intervention responses
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class ImprovedSyntheticPatient:
    """Patient with improved synthetic organ trajectories"""
    patient_id: str
    demographics: Dict
    trajectories: List[Dict]  # List of organ states over time
    health_events: List[Dict]


class ImprovedSyntheticGenerator:
    """
    Generate synthetic organs with strong physiological coupling
    
    Key improvements:
    1. Multivariate sampling for joint organ generation
    2. Stronger cross-organ effect sizes (0.5-1.0 instead of 0.1-0.2)
    3. AR(1) temporal dynamics with mean reversion
    4. Explicit covariance structure
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        
        # Literature-based statistics
        self.organ_stats = {
            'ALT': {'mean': 25, 'std': 10, 'range': (10, 100)},
            'AST': {'mean': 20, 'std': 8, 'range': (10, 80)},
            'WBC': {'mean': 7.0, 'std': 2.0, 'range': (4.0, 11.0)},
            'cognitive': {'mean': 0.85, 'std': 0.12, 'range': (0.3, 1.0)},
            'alcohol': {'mean': 0.3, 'std': 0.2, 'range': (0, 1)},
            'exercise': {'mean': 0.4, 'std': 0.2, 'range': (0, 1)},
            'diet': {'mean': 0.5, 'std': 0.15, 'range': (0, 1)},
        }
        
        # STRONGER effect sizes (increased from 0.15 to 0.5-1.0)
        self.effect_sizes = {
            'glucose_alt': 0.5,        # Was 0.15, now 0.5
            'triglycerides_alt': 0.4,  # Was 0.2, now 0.4
            'bmi_alt': 0.8,            # Was 0.3, now 0.8
            'bmi_wbc': 0.5,            # Was 0.15, now 0.5
            'bp_cognitive': -0.003,    # Was -0.001, now -0.003
            'alcohol_alt': 30.0,       # Was 15, now 30
            'exercise_alt': -10.0,     # Was -3, now -10
        }
        
        # Covariance matrix for joint sampling
        # [ALT, WBC, Cognitive, Alcohol, Exercise]
        self.build_covariance_matrix()
    
    def build_covariance_matrix(self):
        """
        Build covariance matrix for multivariate sampling
        
        This ensures organs are generated jointly with proper correlations
        """
        # Standard deviations
        std_alt = self.organ_stats['ALT']['std']
        std_wbc = self.organ_stats['WBC']['std']
        std_cog = self.organ_stats['cognitive']['std']
        std_alc = self.organ_stats['alcohol']['std']
        std_ex = self.organ_stats['exercise']['std']
        
        # Correlation matrix (from literature)
        corr = np.array([
            [1.0,  0.18, -0.10,  0.25, -0.15],  # ALT
            [0.18, 1.0,  -0.08,  0.10, -0.20],  # WBC
            [-0.10, -0.08, 1.0,  -0.15,  0.25],  # Cognitive
            [0.25, 0.10, -0.15,  1.0,  -0.30],  # Alcohol
            [-0.15, -0.20, 0.25, -0.30,  1.0]   # Exercise
        ])
        
        # Convert to covariance
        std_diag = np.diag([std_alt, std_wbc, std_cog, std_alc, std_ex])
        self.cov_matrix = std_diag @ corr @ std_diag
    
    def generate_baseline_multivariate(
        self,
        real_organs: Dict,
        demographics: Dict
    ) -> Dict:
        """
        Generate synthetic organs using multivariate sampling
        
        This ensures proper joint distribution with correlations
        """
        # Extract conditioning variables
        glucose = real_organs.get('glucose', 100)
        triglycerides = real_organs.get('triglycerides', 150)
        systolic_bp = real_organs.get('systolic_bp', 120)
        bmi = demographics.get('bmi', 27)
        age = demographics.get('age', 50)
        
        # Compute conditional means (STRONGER effects)
        mean_alt = self.organ_stats['ALT']['mean']
        mean_alt += self.effect_sizes['glucose_alt'] * (glucose - 100)
        mean_alt += self.effect_sizes['triglycerides_alt'] * (triglycerides - 150) / 50
        mean_alt += self.effect_sizes['bmi_alt'] * (bmi - 25)
        if age > 50:
            mean_alt += 0.3 * (age - 50)  # Stronger age effect
        
        mean_wbc = self.organ_stats['WBC']['mean']
        mean_wbc += self.effect_sizes['bmi_wbc'] * (bmi - 25)
        if glucose > 126:  # Diabetes
            mean_wbc += 1.5  # Stronger inflammation
        
        mean_cognitive = self.organ_stats['cognitive']['mean']
        mean_cognitive += self.effect_sizes['bp_cognitive'] * (systolic_bp - 120)
        if age > 60:
            mean_cognitive -= 0.003 * (age - 60)  # Stronger age decline
        
        mean_alcohol = self.organ_stats['alcohol']['mean']
        mean_exercise = self.organ_stats['exercise']['mean']
        
        # Sample from multivariate normal
        mean_vector = np.array([mean_alt, mean_wbc, mean_cognitive, mean_alcohol, mean_exercise])
        
        try:
            samples = multivariate_normal(mean_vector, self.cov_matrix).rvs()
        except:
            # Fallback if covariance is not positive definite
            samples = mean_vector + np.random.randn(5) * np.sqrt(np.diag(self.cov_matrix))
        
        # Unpack and clip to valid ranges
        alt = np.clip(samples[0], *self.organ_stats['ALT']['range'])
        wbc = np.clip(samples[1], *self.organ_stats['WBC']['range'])
        cognitive = np.clip(samples[2], *self.organ_stats['cognitive']['range'])
        alcohol = np.clip(samples[3], *self.organ_stats['alcohol']['range'])
        exercise = np.clip(samples[4], *self.organ_stats['exercise']['range'])
        
        # AST follows ALT with ratio
        ast = np.clip(alt * 0.8 + np.random.normal(0, 2), *self.organ_stats['AST']['range'])
        
        # Diet and sleep (less correlated)
        diet = np.clip(np.random.normal(self.organ_stats['diet']['mean'], 
                                       self.organ_stats['diet']['std']), 0, 1)
        sleep = np.clip(np.random.normal(7.0, 1.0), 4, 10)
        
        return {
            'liver': {'ALT': float(alt), 'AST': float(ast)},
            'immune': {'WBC': float(wbc)},
            'neural': {'cognitive_score': float(cognitive)},
            'lifestyle': {
                'alcohol_consumption': float(alcohol),
                'exercise_frequency': float(exercise),
                'diet_quality': float(diet),
                'sleep_hours': float(sleep)
            }
        }
    
    def evolve_temporal_ar1(
        self,
        prev_state: Dict,
        real_organs_t: Dict,
        real_organs_t1: Dict,
        demographics: Dict,
        time_delta_months: int = 6
    ) -> Dict:
        """
        Evolve organs using AR(1) process with mean reversion
        
        X_t+1 = ρ * X_t + (1-ρ) * μ_t + drift + noise
        
        where:
        - ρ = autocorrelation (0.7-0.9, organs have inertia)
        - μ_t = conditional mean given real organs
        - drift = systematic changes (age, interventions)
        - noise = random fluctuations
        """
        time_factor = time_delta_months / 6
        
        # AR(1) parameter (organ inertia)
        rho = 0.8  # Strong autocorrelation
        
        # Extract previous state
        prev_alt = prev_state['liver']['ALT']
        prev_wbc = prev_state['immune']['WBC']
        prev_cog = prev_state['neural']['cognitive_score']
        prev_lifestyle = prev_state['lifestyle']
        
        # Compute new conditional means based on CURRENT real organs
        glucose_t1 = real_organs_t1.get('glucose', 100)
        bp_t1 = real_organs_t1.get('systolic_bp', 120)
        bmi = demographics.get('bmi', 27)
        age = demographics.get('age', 50)
        
        # Target means (where organs want to go)
        target_alt = self.organ_stats['ALT']['mean']
        target_alt += self.effect_sizes['glucose_alt'] * (glucose_t1 - 100)
        target_alt += self.effect_sizes['bmi_alt'] * (bmi - 25)
        
        target_wbc = self.organ_stats['WBC']['mean']
        target_wbc += self.effect_sizes['bmi_wbc'] * (bmi - 25)
        
        target_cog = self.organ_stats['cognitive']['mean']
        target_cog += self.effect_sizes['bp_cognitive'] * (bp_t1 - 120)
        
        # Drift terms (systematic changes)
        drift_alt = 0
        drift_wbc = 0
        drift_cog = 0
        
        # Lifestyle changes (small random walk)
        new_alcohol = prev_lifestyle['alcohol_consumption'] + np.random.normal(0, 0.05 * time_factor)
        new_exercise = prev_lifestyle['exercise_frequency'] + np.random.normal(0, 0.05 * time_factor)
        new_diet = prev_lifestyle['diet_quality'] + np.random.normal(0, 0.05 * time_factor)
        new_sleep = prev_lifestyle['sleep_hours'] + np.random.normal(0, 0.3 * time_factor)
        
        # Lifestyle effects on drift
        alcohol_change = new_alcohol - prev_lifestyle['alcohol_consumption']
        exercise_change = new_exercise - prev_lifestyle['exercise_frequency']
        
        if alcohol_change > 0.1:  # Increased drinking
            drift_alt += self.effect_sizes['alcohol_alt'] * alcohol_change * time_factor
        
        if exercise_change > 0.1:  # Increased exercise
            drift_alt += self.effect_sizes['exercise_alt'] * exercise_change * time_factor
            drift_wbc -= 0.5 * exercise_change * time_factor
            drift_cog += 0.02 * exercise_change * time_factor
        
        # Age effect
        if age > 50:
            drift_alt += 0.2 * time_factor
        if age > 60:
            drift_cog -= 0.002 * time_factor
        
        # AR(1) evolution with mean reversion
        new_alt = rho * prev_alt + (1 - rho) * target_alt + drift_alt + np.random.normal(0, 3 * time_factor)
        new_wbc = rho * prev_wbc + (1 - rho) * target_wbc + drift_wbc + np.random.normal(0, 0.5 * time_factor)
        new_cog = rho * prev_cog + (1 - rho) * target_cog + drift_cog + np.random.normal(0, 0.02 * time_factor)
        
        # Clip to valid ranges
        new_alt = np.clip(new_alt, *self.organ_stats['ALT']['range'])
        new_wbc = np.clip(new_wbc, *self.organ_stats['WBC']['range'])
        new_cog = np.clip(new_cog, *self.organ_stats['cognitive']['range'])
        new_alcohol = np.clip(new_alcohol, 0, 1)
        new_exercise = np.clip(new_exercise, 0, 1)
        new_diet = np.clip(new_diet, 0, 1)
        new_sleep = np.clip(new_sleep, 4, 10)
        
        # AST follows ALT
        new_ast = np.clip(new_alt * 0.8 + np.random.normal(0, 2), *self.organ_stats['AST']['range'])
        
        return {
            'liver': {'ALT': float(new_alt), 'AST': float(new_ast)},
            'immune': {'WBC': float(new_wbc)},
            'neural': {'cognitive_score': float(new_cog)},
            'lifestyle': {
                'alcohol_consumption': float(new_alcohol),
                'exercise_frequency': float(new_exercise),
                'diet_quality': float(new_diet),
                'sleep_hours': float(new_sleep)
            }
        }
    
    def generate_patient_trajectory(
        self,
        patient_id: str,
        real_organs_trajectory: List[Dict],
        demographics: Dict,
        n_timepoints: int = 10
    ) -> ImprovedSyntheticPatient:
        """
        Generate complete patient trajectory with improved dynamics
        """
        trajectories = []
        
        # Baseline (t=0)
        baseline_synthetic = self.generate_baseline_multivariate(
            real_organs_trajectory[0], demographics
        )
        
        baseline_state = {
            'age': demographics['age'],
            'time': 0,
            **baseline_synthetic
        }
        trajectories.append(baseline_state)
        
        # Temporal evolution
        for t in range(1, min(n_timepoints, len(real_organs_trajectory))):
            new_state = self.evolve_temporal_ar1(
                prev_state=trajectories[-1],
                real_organs_t=real_organs_trajectory[t-1],
                real_organs_t1=real_organs_trajectory[t],
                demographics=demographics,
                time_delta_months=6
            )
            
            new_state['age'] = demographics['age'] + t * 0.5
            new_state['time'] = t
            trajectories.append(new_state)
        
        return ImprovedSyntheticPatient(
            patient_id=patient_id,
            demographics=demographics,
            trajectories=trajectories,
            health_events=[]
        )


def compare_generators():
    """Compare old vs new generator"""
    print("="*80)
    print("GENERATOR COMPARISON: Old vs Improved")
    print("="*80)
    
    # Test patient
    real_organs = {
        'glucose': 150,  # High glucose
        'triglycerides': 200,
        'systolic_bp': 140,
    }
    demographics = {'age': 55, 'bmi': 32, 'sex': 'male'}
    
    # Old generator (weak effects)
    print("\nOLD GENERATOR (Weak Effects):")
    old_alt = 25 + 0.15 * (150 - 100) + 0.2 * (200 - 150)/50 + 0.3 * (32 - 25)
    print(f"  ALT = 25 + 0.15*(glucose-100) + 0.2*(trig-150)/50 + 0.3*(bmi-25)")
    print(f"  ALT = {old_alt:.1f} U/L")
    print(f"  Effect of high glucose+trig+BMI: {old_alt - 25:.1f} U/L")
    
    # New generator (strong effects)
    print("\nNEW GENERATOR (Strong Effects):")
    new_alt = 25 + 0.5 * (150 - 100) + 0.4 * (200 - 150)/50 + 0.8 * (32 - 25)
    print(f"  ALT = 25 + 0.5*(glucose-100) + 0.4*(trig-150)/50 + 0.8*(bmi-25)")
    print(f"  ALT = {new_alt:.1f} U/L")
    print(f"  Effect of high glucose+trig+BMI: {new_alt - 25:.1f} U/L")
    
    print(f"\nDifference: {new_alt - old_alt:.1f} U/L ({(new_alt - old_alt)/old_alt * 100:.0f}% stronger)")
    
    # Multivariate sampling test
    print("\n" + "="*80)
    print("MULTIVARIATE SAMPLING TEST")
    print("="*80)
    
    gen = ImprovedSyntheticGenerator()
    
    # Generate 1000 patients
    n_samples = 1000
    alts = []
    wbcs = []
    cogs = []
    
    for _ in range(n_samples):
        synthetic = gen.generate_baseline_multivariate(real_organs, demographics)
        alts.append(synthetic['liver']['ALT'])
        wbcs.append(synthetic['immune']['WBC'])
        cogs.append(synthetic['neural']['cognitive_score'])
    
    alts = np.array(alts)
    wbcs = np.array(wbcs)
    cogs = np.array(cogs)
    
    print(f"\nGenerated {n_samples} patients:")
    print(f"  ALT: mean={np.mean(alts):.1f}, std={np.std(alts):.1f}")
    print(f"  WBC: mean={np.mean(wbcs):.2f}, std={np.std(wbcs):.2f}")
    print(f"  Cognitive: mean={np.mean(cogs):.2f}, std={np.std(cogs):.2f}")
    
    print(f"\nCorrelations:")
    print(f"  ALT-WBC: {np.corrcoef(alts, wbcs)[0,1]:.3f} (target: 0.18)")
    print(f"  ALT-Cognitive: {np.corrcoef(alts, cogs)[0,1]:.3f} (target: -0.10)")
    print(f"  WBC-Cognitive: {np.corrcoef(wbcs, cogs)[0,1]:.3f} (target: -0.08)")


if __name__ == '__main__':
    compare_generators()
