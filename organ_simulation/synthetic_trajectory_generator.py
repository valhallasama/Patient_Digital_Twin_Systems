#!/usr/bin/env python3
"""
Physics-Informed Synthetic Longitudinal Trajectory Generator

Generates realistic organ trajectories for systems lacking real longitudinal data.
Uses medical knowledge + cross-organ correlations to create plausible temporal dynamics.

Strategy: Hybrid approach
- Metabolic/CV/Kidney: Use real NHANES temporal data (33,994 transitions)
- Liver/Immune/Neural/Lifestyle: Generate synthetic trajectories using physics-informed rules

This enables full digital twin functionality while awaiting real cohort access.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class SyntheticPatient:
    """Synthetic patient with longitudinal trajectory"""
    patient_id: str
    demographics: Dict
    trajectories: Dict  # {timepoint: {organ: values}}
    events: List[Dict]  # Health events (diagnosis, intervention)


class PhysicsInformedSyntheticGenerator:
    """
    Generate synthetic organ trajectories using medical knowledge
    
    Key principles:
    1. Cross-organ correlations (glucose ↑ → ALT ↑)
    2. Lifestyle effects (alcohol ↑ → ALT ↑, exercise ↑ → cognitive ↑)
    3. Age-related changes (cognitive decline, organ aging)
    4. Intervention responses (alcohol reduction → ALT improvement)
    5. Physiological noise and individual variation
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.literature_stats = self._load_literature_statistics()
        
    def _load_literature_statistics(self) -> Dict:
        """Population statistics from medical literature"""
        return {
            'liver': {
                'ALT': {
                    'mean': 25, 'std': 10, 'range': (10, 300),
                    'age_correlation': 0.15,
                    'alcohol_correlation': 0.35,
                    'bmi_correlation': 0.28
                },
                'AST': {
                    'mean': 22, 'std': 8, 'range': (10, 250),
                    'ast_alt_ratio': 0.7
                }
            },
            'immune': {
                'WBC': {
                    'mean': 7.0, 'std': 2.0, 'range': (3.0, 15.0),
                    'inflammation_effect': 2.0,
                    'infection_spike': (3.0, 6.0)
                }
            },
            'neural': {
                'cognitive': {
                    'baseline_40yr': 0.95,
                    'decline_rate_per_year': 0.002,
                    'education_bonus': 0.05,
                    'exercise_protection': 0.0002,
                    'vascular_damage': 0.0003
                }
            },
            'lifestyle': {
                'alcohol': {'mean': 0.3, 'std': 0.2, 'range': (0, 1)},
                'exercise': {'mean': 0.4, 'std': 0.2, 'range': (0, 1)},
                'diet': {'mean': 0.5, 'std': 0.15, 'range': (0, 1)},
                'sleep': {'mean': 7.0, 'std': 1.0, 'range': (4, 10)}
            }
        }
    
    def generate_cohort(
        self,
        n_patients: int = 10000,
        n_timepoints: int = 10,
        interval_months: int = 6,
        output_path: Optional[str] = None
    ) -> List[SyntheticPatient]:
        """
        Generate synthetic longitudinal cohort
        
        Args:
            n_patients: Number of patients
            n_timepoints: Time points per patient
            interval_months: Months between measurements
            output_path: Save path (optional)
        
        Returns:
            List of SyntheticPatient objects
        """
        print(f"Generating {n_patients} synthetic patients...")
        print(f"  Timepoints: {n_timepoints} (every {interval_months} months)")
        print(f"  Total duration: {n_timepoints * interval_months / 12:.1f} years")
        
        cohort = []
        for i in range(n_patients):
            patient = self._generate_patient(
                patient_id=f"SYN_{i:06d}",
                n_timepoints=n_timepoints,
                interval_months=interval_months
            )
            cohort.append(patient)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{n_patients} patients")
        
        if output_path:
            self.save_cohort(cohort, output_path)
        
        return cohort
    
    def _generate_patient(
        self,
        patient_id: str,
        n_timepoints: int,
        interval_months: int
    ) -> SyntheticPatient:
        """Generate single patient trajectory"""
        
        # Sample demographics
        demographics = self._sample_demographics()
        
        # Initialize trajectories
        trajectories = {}
        events = []
        
        # Generate baseline (t=0)
        age_baseline = demographics['age']
        trajectories[0] = self._generate_baseline_state(demographics)
        
        # Evolve over time
        for t in range(1, n_timepoints):
            age = age_baseline + (t * interval_months / 12)
            prev_state = trajectories[t-1]
            
            # Check for health events (diagnosis, intervention)
            event = self._check_health_events(prev_state, age, t * interval_months)
            if event:
                events.append(event)
            
            # Evolve state
            new_state = self._evolve_state(
                prev_state=prev_state,
                age=age,
                interval_months=interval_months,
                recent_events=[e for e in events if e['time_months'] >= (t-2) * interval_months]
            )
            
            new_state['time_months'] = t * interval_months
            new_state['age'] = age
            trajectories[t] = new_state
        
        return SyntheticPatient(
            patient_id=patient_id,
            demographics=demographics,
            trajectories=trajectories,
            events=events
        )
    
    def _sample_demographics(self) -> Dict:
        """Sample patient demographics"""
        age = np.random.randint(30, 80)
        gender = np.random.choice(['male', 'female'])
        
        # BMI from realistic distribution
        bmi = np.random.normal(27, 5)
        bmi = np.clip(bmi, 18, 45)
        
        # Education (affects cognitive baseline)
        education_years = np.random.choice([12, 14, 16, 18, 20], p=[0.2, 0.3, 0.25, 0.15, 0.1])
        
        # Genetic risk factors (simplified)
        genetic_risk = {
            'metabolic': np.random.uniform(0, 1),
            'cardiovascular': np.random.uniform(0, 1),
            'liver': np.random.uniform(0, 1),
            'neural': np.random.uniform(0, 1)
        }
        
        return {
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'education_years': education_years,
            'genetic_risk': genetic_risk
        }
    
    def _generate_baseline_state(self, demographics: Dict) -> Dict:
        """Generate baseline organ states"""
        age = demographics['age']
        bmi = demographics['bmi']
        education = demographics['education_years']
        
        # Sample baseline lifestyle
        alcohol = np.abs(np.random.normal(0.3, 0.2))
        exercise = np.abs(np.random.normal(0.4, 0.2))
        diet = np.abs(np.random.normal(0.5, 0.15))
        
        alcohol = np.clip(alcohol, 0, 1)
        exercise = np.clip(exercise, 0, 1)
        diet = np.clip(diet, 0, 1)
        
        # Liver baseline (physics-informed)
        alt_base = self.literature_stats['liver']['ALT']['mean']
        
        # Alcohol effect on baseline (calibrated to match literature correlation)
        if alcohol > 0.6:
            alt_base += (alcohol - 0.6) * 25
        
        # BMI effect (fatty liver) (calibrated)
        if bmi > 28:
            alt_base += (bmi - 28) * 0.6
        
        # Age effect (calibrated)
        if age > 55:
            alt_base += (age - 55) * 0.15
        
        # Genetic risk (calibrated)
        alt_base += (demographics['genetic_risk']['liver'] - 0.5) * 8
        
        # Add noise (increased to match literature std)
        alt = alt_base + np.random.normal(0, 8)
        alt = np.clip(alt, *self.literature_stats['liver']['ALT']['range'])
        
        # AST follows ALT with ratio
        ast_ratio = self.literature_stats['liver']['AST']['ast_alt_ratio']
        ast = alt * np.random.normal(ast_ratio, 0.1)
        ast = np.clip(ast, *self.literature_stats['liver']['AST']['range'])
        
        # Immune baseline
        wbc_base = self.literature_stats['immune']['WBC']['mean']
        
        # Chronic inflammation from metabolic syndrome
        if bmi > 30:
            wbc_base += 1.2
        if alcohol > 0.7:
            wbc_base += 0.8
        
        wbc = wbc_base + np.random.normal(0, 1.0)
        wbc = np.clip(wbc, *self.literature_stats['immune']['WBC']['range'])
        
        # Neural baseline
        cog_stats = self.literature_stats['neural']['cognitive']
        cog_base = cog_stats['baseline_40yr'] - (age - 40) * cog_stats['decline_rate_per_year']
        
        # Education effect
        if education > 16:
            cog_base += cog_stats['education_bonus']
        
        # Vascular health effect
        if bmi > 30:
            cog_base -= 0.03
        
        cognitive = cog_base + np.random.normal(0, 0.05)
        cognitive = np.clip(cognitive, 0.3, 1.0)
        
        return {
            'time_months': 0,
            'age': age,
            'liver': {'ALT': alt, 'AST': ast},
            'immune': {'WBC': wbc},
            'neural': {'cognitive_score': cognitive},
            'lifestyle': {
                'alcohol_consumption': alcohol,
                'exercise_frequency': exercise,
                'diet_quality': diet,
                'sleep_hours': np.random.normal(7, 1)
            }
        }
    
    def _evolve_state(
        self,
        prev_state: Dict,
        age: float,
        interval_months: int,
        recent_events: List[Dict]
    ) -> Dict:
        """Evolve organ states using physics-informed rules"""
        
        # Extract previous values
        prev_liver = prev_state['liver']
        prev_immune = prev_state['immune']
        prev_neural = prev_state['neural']
        prev_lifestyle = prev_state['lifestyle']
        
        # Evolve lifestyle (people's habits change!)
        new_lifestyle = self._evolve_lifestyle(
            prev_lifestyle, prev_liver, age, recent_events
        )
        
        # Evolve liver
        new_liver = self._evolve_liver(
            prev_liver, new_lifestyle, prev_lifestyle, age, interval_months
        )
        
        # Evolve immune
        new_immune = self._evolve_immune(
            prev_immune, new_liver, new_lifestyle, interval_months
        )
        
        # Evolve neural
        new_neural = self._evolve_neural(
            prev_neural, new_liver, new_lifestyle, age, interval_months
        )
        
        return {
            'liver': new_liver,
            'immune': new_immune,
            'neural': new_neural,
            'lifestyle': new_lifestyle
        }
    
    def _evolve_lifestyle(
        self,
        prev_lifestyle: Dict,
        prev_liver: Dict,
        age: float,
        recent_events: List[Dict]
    ) -> Dict:
        """Evolve lifestyle factors (they change over time!)"""
        
        alcohol = prev_lifestyle['alcohol_consumption']
        exercise = prev_lifestyle['exercise_frequency']
        diet = prev_lifestyle['diet_quality']
        
        # Base random walk
        delta_alcohol = np.random.normal(0, 0.05)
        delta_exercise = np.random.normal(0, 0.05)
        delta_diet = np.random.normal(0, 0.05)
        
        # Health event triggers behavior change
        for event in recent_events:
            if event['type'] == 'liver_disease_diagnosis':
                delta_alcohol -= 0.3  # Reduce drinking
                delta_diet += 0.2     # Improve diet
            elif event['type'] == 'cardiovascular_event':
                delta_exercise += 0.25  # Increase exercise
                delta_diet += 0.15
            elif event['type'] == 'diabetes_diagnosis':
                delta_diet += 0.2
                delta_exercise += 0.15
        
        # Elevated liver enzymes trigger behavior change
        if prev_liver['ALT'] > 60:
            delta_alcohol -= 0.1
            delta_diet += 0.08
        
        # Age effects
        if age > 65:
            delta_alcohol -= 0.02  # Older people drink less
            delta_exercise -= 0.03  # Exercise decreases with age
        
        # Apply changes
        new_alcohol = np.clip(alcohol + delta_alcohol, 0, 1)
        new_exercise = np.clip(exercise + delta_exercise, 0, 1)
        new_diet = np.clip(diet + delta_diet, 0, 1)
        
        return {
            'alcohol_consumption': new_alcohol,
            'exercise_frequency': new_exercise,
            'diet_quality': new_diet,
            'sleep_hours': np.random.normal(7, 1)
        }
    
    def _evolve_liver(
        self,
        prev_liver: Dict,
        new_lifestyle: Dict,
        prev_lifestyle: Dict,
        age: float,
        interval_months: int
    ) -> Dict:
        """Evolve liver biomarkers using physics-informed rules"""
        
        prev_alt = prev_liver['ALT']
        delta_alt = 0
        
        time_factor = interval_months / 6  # Normalize to 6-month intervals
        
        # Alcohol effect (strengthened to improve correlation)
        alcohol = new_lifestyle['alcohol_consumption']
        if alcohol > 0.7:
            delta_alt += (alcohol - 0.5) * 4.5 * time_factor  # Heavy drinking
        elif alcohol > 0.5:
            delta_alt += (alcohol - 0.5) * 2.5 * time_factor  # Moderate
        
        # Recovery from alcohol reduction
        alcohol_reduction = prev_lifestyle['alcohol_consumption'] - alcohol
        if alcohol_reduction > 0.1 and prev_alt > 40:
            delta_alt -= alcohol_reduction * 4.0 * time_factor
        
        # Diet and exercise effects
        if new_lifestyle['exercise_frequency'] > 0.6:
            delta_alt -= 0.6 * time_factor
        if new_lifestyle['diet_quality'] > 0.6:
            delta_alt -= 0.4 * time_factor
        
        # Age-related increase
        if age > 50:
            delta_alt += 0.15 * time_factor
        
        # Physiological noise
        delta_alt += np.random.normal(0, 2.5)
        
        # Apply change
        new_alt = prev_alt + delta_alt
        new_alt = np.clip(new_alt, *self.literature_stats['liver']['ALT']['range'])
        
        # AST follows ALT
        ast_ratio = self.literature_stats['liver']['AST']['ast_alt_ratio']
        new_ast = new_alt * np.random.normal(ast_ratio, 0.08)
        new_ast = np.clip(new_ast, *self.literature_stats['liver']['AST']['range'])
        
        return {'ALT': new_alt, 'AST': new_ast}
    
    def _evolve_immune(
        self,
        prev_immune: Dict,
        new_liver: Dict,
        new_lifestyle: Dict,
        interval_months: int
    ) -> Dict:
        """Evolve immune markers"""
        
        prev_wbc = prev_immune['WBC']
        delta_wbc = 0
        
        time_factor = interval_months / 6
        
        # Random infection events
        if np.random.random() < 0.04 * time_factor:
            spike_range = self.literature_stats['immune']['WBC']['infection_spike']
            delta_wbc += np.random.uniform(*spike_range)
        
        # Chronic inflammation from liver disease
        if new_liver['ALT'] > 60:
            delta_wbc += 0.6 * time_factor
        
        # Exercise reduces inflammation
        if new_lifestyle['exercise_frequency'] > 0.6:
            delta_wbc -= 0.4 * time_factor
        
        # Physiological noise
        delta_wbc += np.random.normal(0, 0.6)
        
        new_wbc = prev_wbc + delta_wbc
        new_wbc = np.clip(new_wbc, *self.literature_stats['immune']['WBC']['range'])
        
        return {'WBC': new_wbc}
    
    def _evolve_neural(
        self,
        prev_neural: Dict,
        new_liver: Dict,
        new_lifestyle: Dict,
        age: float,
        interval_months: int
    ) -> Dict:
        """Evolve cognitive function"""
        
        prev_cog = prev_neural['cognitive_score']
        delta_cog = 0
        
        time_factor = interval_months / 6
        cog_stats = self.literature_stats['neural']['cognitive']
        
        # Age-related decline
        if age > 60:
            delta_cog -= cog_stats['decline_rate_per_year'] * (age - 60) * time_factor / 2
        
        # Lifestyle protective effects
        if new_lifestyle['exercise_frequency'] > 0.6:
            delta_cog += cog_stats['exercise_protection'] * new_lifestyle['exercise_frequency'] * time_factor
        
        if new_lifestyle['diet_quality'] > 0.6:
            delta_cog += 0.0001 * new_lifestyle['diet_quality'] * time_factor
        
        # Vascular damage from liver disease
        if new_liver['ALT'] > 80:
            delta_cog -= cog_stats['vascular_damage'] * time_factor
        
        # Physiological noise
        delta_cog += np.random.normal(0, 0.01)
        
        new_cog = prev_cog + delta_cog
        new_cog = np.clip(new_cog, 0.3, 1.0)
        
        return {'cognitive_score': new_cog}
    
    def _check_health_events(
        self,
        state: Dict,
        age: float,
        time_months: int
    ) -> Optional[Dict]:
        """Check for health events (diagnosis, intervention)"""
        
        # Liver disease diagnosis
        if state['liver']['ALT'] > 80 and np.random.random() < 0.1:
            return {
                'type': 'liver_disease_diagnosis',
                'time_months': time_months,
                'age': age,
                'trigger': f"ALT={state['liver']['ALT']:.1f}"
            }
        
        # Cardiovascular event (simplified)
        if age > 60 and np.random.random() < 0.02:
            return {
                'type': 'cardiovascular_event',
                'time_months': time_months,
                'age': age
            }
        
        # Diabetes diagnosis
        if np.random.random() < 0.01:
            return {
                'type': 'diabetes_diagnosis',
                'time_months': time_months,
                'age': age
            }
        
        return None
    
    def validate_cohort(self, cohort: List[SyntheticPatient]) -> Dict:
        """Validate synthetic cohort against literature statistics"""
        
        print("\n" + "="*80)
        print("VALIDATION AGAINST LITERATURE STATISTICS")
        print("="*80)
        
        # Extract baseline values
        alt_values = [p.trajectories[0]['liver']['ALT'] for p in cohort]
        wbc_values = [p.trajectories[0]['immune']['WBC'] for p in cohort]
        cog_values = [p.trajectories[0]['neural']['cognitive_score'] for p in cohort]
        ages = [p.demographics['age'] for p in cohort]
        alcohols = [p.trajectories[0]['lifestyle']['alcohol_consumption'] for p in cohort]
        
        validation = {}
        
        # ALT statistics
        alt_mean = np.mean(alt_values)
        alt_std = np.std(alt_values)
        alt_expected = self.literature_stats['liver']['ALT']
        
        validation['ALT'] = {
            'mean': alt_mean,
            'std': alt_std,
            'expected_mean': alt_expected['mean'],
            'expected_std': alt_expected['std'],
            'pass': abs(alt_mean - alt_expected['mean']) < 3 and abs(alt_std - alt_expected['std']) < 3
        }
        
        # WBC statistics
        wbc_mean = np.mean(wbc_values)
        wbc_std = np.std(wbc_values)
        wbc_expected = self.literature_stats['immune']['WBC']
        
        validation['WBC'] = {
            'mean': wbc_mean,
            'std': wbc_std,
            'expected_mean': wbc_expected['mean'],
            'expected_std': wbc_expected['std'],
            'pass': abs(wbc_mean - wbc_expected['mean']) < 0.5 and abs(wbc_std - wbc_expected['std']) < 0.5
        }
        
        # Cognitive statistics
        cog_mean = np.mean(cog_values)
        cog_std = np.std(cog_values)
        
        validation['Cognitive'] = {
            'mean': cog_mean,
            'std': cog_std,
            'pass': 0.75 < cog_mean < 0.95
        }
        
        # Correlations
        alt_age_corr = np.corrcoef(alt_values, ages)[0, 1]
        alt_alcohol_corr = np.corrcoef(alt_values, alcohols)[0, 1]
        
        alt_expected_corr = self.literature_stats['liver']['ALT']
        
        validation['Correlations'] = {
            'ALT_age': alt_age_corr,
            'ALT_age_expected': alt_expected_corr['age_correlation'],
            'ALT_alcohol': alt_alcohol_corr,
            'ALT_alcohol_expected': alt_expected_corr['alcohol_correlation'],
            'pass': (abs(alt_age_corr - alt_expected_corr['age_correlation']) < 0.1 and
                    abs(alt_alcohol_corr - alt_expected_corr['alcohol_correlation']) < 0.15)
        }
        
        # Print results
        for metric, values in validation.items():
            if metric != 'Correlations':
                print(f"\n{metric}:")
                print(f"  Generated: mean={values['mean']:.2f}, std={values['std']:.2f}")
                if 'expected_mean' in values:
                    print(f"  Expected:  mean={values['expected_mean']:.2f}, std={values['expected_std']:.2f}")
                print(f"  {'✓ PASS' if values['pass'] else '✗ FAIL'}")
        
        print(f"\nCorrelations:")
        corr = validation['Correlations']
        print(f"  ALT-Age: {corr['ALT_age']:.3f} (expected ~{corr['ALT_age_expected']:.2f})")
        print(f"  ALT-Alcohol: {corr['ALT_alcohol']:.3f} (expected ~{corr['ALT_alcohol_expected']:.2f})")
        print(f"  {'✓ PASS' if corr['pass'] else '✗ FAIL'}")
        
        return validation
    
    def save_cohort(self, cohort: List[SyntheticPatient], filepath: str):
        """Save synthetic cohort"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(cohort, f)
        
        print(f"\n✓ Saved {len(cohort)} patients to {filepath}")


def main():
    """Generate and validate synthetic cohort"""
    print("="*80)
    print("PHYSICS-INFORMED SYNTHETIC LONGITUDINAL COHORT GENERATION")
    print("="*80)
    print("\nStrategy: Generate realistic organ trajectories for missing systems")
    print("  ✅ Metabolic/CV/Kidney: Use real NHANES data (33,994 transitions)")
    print("  🔬 Liver/Immune/Neural/Lifestyle: Generate synthetic (physics-informed)")
    
    generator = PhysicsInformedSyntheticGenerator(seed=42)
    
    # Generate cohort
    print("\n" + "="*80)
    print("GENERATING COHORT")
    print("="*80)
    
    cohort = generator.generate_cohort(
        n_patients=10000,
        n_timepoints=10,
        interval_months=6,
        output_path='./data/synthetic_longitudinal_cohort.pkl'
    )
    
    # Validate
    validation = generator.validate_cohort(cohort)
    
    # Show example trajectory
    print("\n" + "="*80)
    print("EXAMPLE PATIENT TRAJECTORY")
    print("="*80)
    
    example = cohort[0]
    print(f"\nPatient: {example.patient_id}")
    print(f"Demographics:")
    print(f"  Age: {example.demographics['age']} years")
    print(f"  Gender: {example.demographics['gender']}")
    print(f"  BMI: {example.demographics['bmi']:.1f}")
    print(f"  Education: {example.demographics['education_years']} years")
    
    print(f"\nTrajectory (10 timepoints over 5 years):")
    print(f"{'Time':<8} {'ALT':<8} {'WBC':<8} {'Cognitive':<10} {'Alcohol':<8} {'Exercise':<8}")
    print("-" * 65)
    
    for t in range(10):
        tp = example.trajectories[t]
        print(f"{tp['time_months']:<8} "
              f"{tp['liver']['ALT']:<8.1f} "
              f"{tp['immune']['WBC']:<8.1f} "
              f"{tp['neural']['cognitive_score']:<10.3f} "
              f"{tp['lifestyle']['alcohol_consumption']:<8.2f} "
              f"{tp['lifestyle']['exercise_frequency']:<8.2f}")
    
    if example.events:
        print(f"\nHealth Events:")
        for event in example.events:
            print(f"  - {event['type']} at {event['time_months']} months (age {event['age']:.1f})")
    
    print("\n" + "="*80)
    print("✓ SYNTHETIC COHORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Train temporal models on hybrid real+synthetic data")
    print(f"2. Integrate into digital twin system")
    print(f"3. Validate plausibility vs medical literature")
    print(f"4. Document methodology for publication")


if __name__ == '__main__':
    main()
