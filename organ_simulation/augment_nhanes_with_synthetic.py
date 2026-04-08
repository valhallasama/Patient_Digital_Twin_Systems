#!/usr/bin/env python3
"""
Augment NHANES Dataset with Synthetic Missing Organs

Strategy: For each NHANES patient with real metabolic/CV/kidney data,
generate plausible liver/immune/neural/lifestyle values that are:
1. Conditioned on their real organ states
2. Respect known medical correlations
3. Evolve realistically over time

This creates a complete 135K patient dataset with all 7 organs.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent))
from correlation_regularizer import CorrelationRegularizer


@dataclass
class AugmentedPatient:
    """NHANES patient with augmented missing organs"""
    patient_id: str
    real_organs: Dict  # Metabolic, CV, Kidney from NHANES
    synthetic_organs: Dict  # Liver, Immune, Neural, Lifestyle (generated)
    demographics: Dict
    timepoint: int


class ConditionalOrganGenerator:
    """
    Generate missing organ values conditioned on real NHANES data
    
    Key principle: Use real organ states to inform synthetic generation
    Example: High glucose → higher ALT (metabolic-liver coupling)
    """
    
    def __init__(self):
        # Literature-based correlation coefficients
        self.correlations = {
            # Metabolic-Liver
            'glucose_alt': 0.25,      # Hyperglycemia → liver stress
            'triglycerides_alt': 0.30,  # Dyslipidemia → fatty liver
            
            # Cardiovascular-Neural
            'systolic_bp_cognitive': -0.20,  # Hypertension → cognitive decline
            
            # Metabolic-Immune
            'glucose_wbc': 0.15,      # Hyperglycemia → inflammation
            
            # BMI effects
            'bmi_alt': 0.35,          # Obesity → fatty liver
            'bmi_wbc': 0.25,          # Obesity → chronic inflammation
        }
        
        # Population statistics for initialization
        self.population_stats = {
            'ALT': {'mean': 25, 'std': 10, 'range': (10, 100)},
            'AST': {'mean': 23, 'std': 8, 'range': (10, 80)},
            'WBC': {'mean': 7.0, 'std': 2.0, 'range': (4.0, 11.0)},
            'cognitive': {'mean': 0.85, 'std': 0.12, 'range': (0.3, 1.0)},
            'alcohol': {'mean': 0.3, 'std': 0.2, 'range': (0, 1)},
            'exercise': {'mean': 0.4, 'std': 0.2, 'range': (0, 1)},
            'diet': {'mean': 0.5, 'std': 0.15, 'range': (0, 1)},
        }
    
    def generate_liver_conditioned(
        self,
        real_organs: Dict,
        demographics: Dict,
        lifestyle: Optional[Dict] = None
    ) -> Dict:
        """
        Generate liver biomarkers conditioned on real metabolic/CV data
        
        Args:
            real_organs: Real NHANES data (glucose, triglycerides, etc.)
            demographics: Age, BMI, gender
            lifestyle: If already generated, use it; else estimate
        
        Returns:
            {'ALT': float, 'AST': float}
        """
        # Start with population baseline
        alt_base = self.population_stats['ALT']['mean']
        
        # Condition on real metabolic state
        glucose = real_organs.get('glucose', 100)
        triglycerides = real_organs.get('triglycerides', 150)
        
        # Glucose effect (metabolic-liver coupling)
        if glucose > 100:  # Pre-diabetes/diabetes
            glucose_effect = (glucose - 100) / 100 * 15  # Scale effect
            alt_base += glucose_effect * self.correlations['glucose_alt'] / 0.25
        
        # Triglycerides effect (dyslipidemia → fatty liver)
        if triglycerides > 150:
            trig_effect = (triglycerides - 150) / 150 * 12
            alt_base += trig_effect * self.correlations['triglycerides_alt'] / 0.30
        
        # BMI effect (obesity → NAFLD)
        bmi = demographics.get('bmi', 27)
        if bmi > 25:
            bmi_effect = (bmi - 25) * 0.8
            alt_base += bmi_effect * self.correlations['bmi_alt'] / 0.35
        
        # Age effect
        age = demographics.get('age', 50)
        if age > 50:
            alt_base += (age - 50) * 0.15
        
        # Lifestyle effect (if available)
        if lifestyle:
            alcohol = lifestyle.get('alcohol_consumption', 0.3)
            exercise = lifestyle.get('exercise_frequency', 0.4)
            
            if alcohol > 0.6:
                alt_base += (alcohol - 0.6) * 25
            
            if exercise > 0.6:
                alt_base -= (exercise - 0.6) * 8  # Protective effect
        
        # Add individual variation
        alt = alt_base + np.random.normal(0, self.population_stats['ALT']['std'])
        alt = np.clip(alt, *self.population_stats['ALT']['range'])
        
        # AST follows ALT with ratio
        ast_ratio = 0.8 + np.random.normal(0, 0.1)
        ast = alt * ast_ratio
        ast = np.clip(ast, *self.population_stats['AST']['range'])
        
        return {
            'ALT': float(alt),
            'AST': float(ast)
        }
    
    def generate_immune_conditioned(
        self,
        real_organs: Dict,
        demographics: Dict,
        liver: Dict
    ) -> Dict:
        """
        Generate immune markers conditioned on real data and generated liver
        
        Cross-organ coupling:
        - High glucose → inflammation
        - High ALT → liver inflammation
        - High BMI → chronic inflammation
        """
        wbc_base = self.population_stats['WBC']['mean']
        
        # Glucose effect (metabolic-immune coupling)
        glucose = real_organs.get('glucose', 100)
        if glucose > 126:  # Diabetes
            glucose_effect = (glucose - 126) / 100 * 1.5
            wbc_base += glucose_effect * self.correlations['glucose_wbc'] / 0.15
        
        # Liver inflammation coupling
        alt = liver['ALT']
        if alt > 40:  # Elevated liver enzymes
            liver_inflammation = (alt - 40) / 40 * 1.2
            wbc_base += liver_inflammation
        
        # BMI effect (obesity → chronic inflammation)
        bmi = demographics.get('bmi', 27)
        if bmi > 30:
            bmi_effect = (bmi - 30) * 0.15
            wbc_base += bmi_effect * self.correlations['bmi_wbc'] / 0.25
        
        # Age effect
        age = demographics.get('age', 50)
        if age > 65:
            wbc_base += (age - 65) * 0.02  # Immunosenescence
        
        # Random infection events (5% chance)
        if np.random.random() < 0.05:
            wbc_base += np.random.uniform(2.0, 5.0)
        
        # Individual variation
        wbc = wbc_base + np.random.normal(0, self.population_stats['WBC']['std'])
        wbc = np.clip(wbc, *self.population_stats['WBC']['range'])
        
        return {'WBC': float(wbc)}
    
    def generate_neural_conditioned(
        self,
        real_organs: Dict,
        demographics: Dict,
        lifestyle: Dict
    ) -> Dict:
        """
        Generate cognitive score conditioned on cardiovascular health
        
        Cross-organ coupling:
        - High BP → vascular cognitive impairment
        - Exercise → neuroprotection
        - Age → natural decline
        """
        cognitive_base = self.population_stats['cognitive']['mean']
        
        # Age-related decline
        age = demographics.get('age', 50)
        if age > 40:
            age_decline = (age - 40) * 0.002
            cognitive_base -= age_decline
        
        # Cardiovascular-neural coupling
        systolic_bp = real_organs.get('systolic_bp', 120)
        if systolic_bp > 140:  # Hypertension
            bp_effect = (systolic_bp - 140) / 40 * 0.15
            cognitive_base -= bp_effect * abs(self.correlations['systolic_bp_cognitive']) / 0.20
        
        # Cholesterol effect (vascular health)
        total_chol = real_organs.get('total_cholesterol', 200)
        if total_chol > 240:
            chol_effect = (total_chol - 240) / 100 * 0.08
            cognitive_base -= chol_effect
        
        # Education (cognitive reserve)
        education = demographics.get('education', 12)
        if education > 16:
            cognitive_base += 0.05
        
        # Lifestyle protection
        exercise = lifestyle.get('exercise_frequency', 0.4)
        if exercise > 0.6:
            cognitive_base += (exercise - 0.6) * 0.15
        
        diet = lifestyle.get('diet_quality', 0.5)
        if diet > 0.6:
            cognitive_base += (diet - 0.6) * 0.08
        
        # Individual variation
        cognitive = cognitive_base + np.random.normal(0, self.population_stats['cognitive']['std'])
        cognitive = np.clip(cognitive, *self.population_stats['cognitive']['range'])
        
        return {'cognitive_score': float(cognitive)}
    
    def generate_lifestyle_conditioned(
        self,
        real_organs: Dict,
        demographics: Dict
    ) -> Dict:
        """
        Generate lifestyle factors with some correlation to health status
        
        Principle: Unhealthier metabolic state → slightly worse lifestyle
        (Not deterministic - just statistical tendency)
        """
        # Base lifestyle from population
        alcohol = np.abs(np.random.normal(
            self.population_stats['alcohol']['mean'],
            self.population_stats['alcohol']['std']
        ))
        exercise = np.abs(np.random.normal(
            self.population_stats['exercise']['mean'],
            self.population_stats['exercise']['std']
        ))
        diet = np.abs(np.random.normal(
            self.population_stats['diet']['mean'],
            self.population_stats['diet']['std']
        ))
        
        # Weak correlation with metabolic health
        glucose = real_organs.get('glucose', 100)
        bmi = demographics.get('bmi', 27)
        
        # Higher glucose/BMI → slightly worse lifestyle (statistical tendency)
        if glucose > 126:  # Diabetes
            exercise -= 0.05  # Slight tendency toward less exercise
            diet -= 0.05
        
        if bmi > 30:  # Obesity
            exercise -= 0.08
            diet -= 0.08
            alcohol += 0.05
        
        # Age effect on lifestyle
        age = demographics.get('age', 50)
        if age > 65:
            alcohol -= 0.1  # Older adults drink less on average
            exercise -= 0.1  # Less active
        
        # Clip to valid range
        alcohol = np.clip(alcohol, *self.population_stats['alcohol']['range'])
        exercise = np.clip(exercise, *self.population_stats['exercise']['range'])
        diet = np.clip(diet, *self.population_stats['diet']['range'])
        
        sleep_hours = np.clip(np.random.normal(7.0, 1.0), 4, 10)
        
        return {
            'alcohol_consumption': float(alcohol),
            'exercise_frequency': float(exercise),
            'diet_quality': float(diet),
            'sleep_hours': float(sleep_hours)
        }
    
    def generate_temporal_evolution(
        self,
        prev_state: Dict,
        real_organs_t: Dict,
        real_organs_t1: Dict,
        demographics: Dict,
        time_delta_months: int
    ) -> Dict:
        """
        Generate temporal evolution of synthetic organs based on real organ changes
        
        Key: If real organs change (e.g., glucose increases), synthetic organs
        should respond accordingly (e.g., ALT increases)
        """
        time_factor = time_delta_months / 6
        
        # Extract previous synthetic state
        prev_liver = prev_state['liver']
        prev_immune = prev_state['immune']
        prev_neural = prev_state['neural']
        prev_lifestyle = prev_state['lifestyle']
        
        # Detect changes in real organs
        glucose_change = real_organs_t1.get('glucose', 100) - real_organs_t.get('glucose', 100)
        bp_change = real_organs_t1.get('systolic_bp', 120) - real_organs_t.get('systolic_bp', 120)
        
        # Evolve liver
        delta_alt = 0
        
        # Response to glucose change
        if glucose_change > 10:  # Glucose increased
            delta_alt += glucose_change * 0.15 * time_factor
        elif glucose_change < -10:  # Glucose improved
            delta_alt -= abs(glucose_change) * 0.10 * time_factor
        
        # Lifestyle evolution (small random walk)
        new_lifestyle = {
            'alcohol_consumption': prev_lifestyle['alcohol_consumption'] + np.random.normal(0, 0.05),
            'exercise_frequency': prev_lifestyle['exercise_frequency'] + np.random.normal(0, 0.05),
            'diet_quality': prev_lifestyle['diet_quality'] + np.random.normal(0, 0.05),
            'sleep_hours': prev_lifestyle['sleep_hours'] + np.random.normal(0, 0.3)
        }
        
        # Clip lifestyle
        new_lifestyle['alcohol_consumption'] = np.clip(new_lifestyle['alcohol_consumption'], 0, 1)
        new_lifestyle['exercise_frequency'] = np.clip(new_lifestyle['exercise_frequency'], 0, 1)
        new_lifestyle['diet_quality'] = np.clip(new_lifestyle['diet_quality'], 0, 1)
        new_lifestyle['sleep_hours'] = np.clip(new_lifestyle['sleep_hours'], 4, 10)
        
        # Lifestyle effect on liver
        alcohol_change = new_lifestyle['alcohol_consumption'] - prev_lifestyle['alcohol_consumption']
        if alcohol_change > 0.1:
            delta_alt += alcohol_change * 20 * time_factor
        
        exercise_change = new_lifestyle['exercise_frequency'] - prev_lifestyle['exercise_frequency']
        if exercise_change > 0.1:
            delta_alt -= exercise_change * 8 * time_factor
        
        # Age effect
        delta_alt += 0.15 * time_factor
        
        # Noise
        delta_alt += np.random.normal(0, 2.5)
        
        new_alt = np.clip(prev_liver['ALT'] + delta_alt, 10, 100)
        new_ast = np.clip(new_alt * 0.8, 10, 80)
        
        new_liver = {'ALT': float(new_alt), 'AST': float(new_ast)}
        
        # Evolve immune (coupled to liver)
        delta_wbc = 0
        
        if new_alt > prev_liver['ALT'] + 5:  # Liver inflammation increasing
            delta_wbc += 0.5 * time_factor
        
        # Random infection
        if np.random.random() < 0.05:
            delta_wbc += np.random.uniform(2.0, 4.0)
        
        delta_wbc += np.random.normal(0, 0.6)
        
        new_wbc = np.clip(prev_immune['WBC'] + delta_wbc, 4.0, 11.0)
        new_immune = {'WBC': float(new_wbc)}
        
        # Evolve neural (coupled to BP)
        delta_cognitive = 0
        
        # Age decline
        age = demographics.get('age', 50)
        if age > 60:
            delta_cognitive -= 0.001 * (age - 60) * time_factor
        
        # BP effect
        if bp_change > 10:  # BP increased
            delta_cognitive -= 0.002 * time_factor
        
        # Exercise protection
        if new_lifestyle['exercise_frequency'] > 0.6:
            delta_cognitive += 0.0002 * time_factor
        
        delta_cognitive += np.random.normal(0, 0.01)
        
        new_cognitive = np.clip(prev_neural['cognitive_score'] + delta_cognitive, 0.3, 1.0)
        new_neural = {'cognitive_score': float(new_cognitive)}
        
        return {
            'liver': new_liver,
            'immune': new_immune,
            'neural': new_neural,
            'lifestyle': new_lifestyle
        }


class NHANESAugmenter:
    """
    Augment NHANES dataset with synthetic missing organs
    """
    
    def __init__(self):
        self.generator = ConditionalOrganGenerator()
        
        # Target cross-organ correlations from literature
        self.target_correlations = {
            'glucose_alt': 0.25,
            'triglycerides_alt': 0.30,
            'bmi_alt': 0.35,
            'bmi_wbc': 0.25,
            'systolic_bp_cognitive': -0.20,
            'alt_wbc': 0.18  # Liver-immune coupling
        }
    
    def load_nhanes_data(self, nhanes_path: str) -> List[Dict]:
        """Load original NHANES dataset"""
        print(f"Loading NHANES data from {nhanes_path}...")
        
        with open(nhanes_path, 'rb') as f:
            data = pickle.load(f)
        
        patients = data['patients']
        print(f"  ✓ Loaded {len(patients)} NHANES patients")
        
        return patients
    
    def augment_patient_baseline(self, nhanes_patient: Dict) -> Dict:
        """
        Augment single NHANES patient with synthetic organs at baseline
        
        Args:
            nhanes_patient: Original NHANES data with metabolic/CV/kidney
        
        Returns:
            Augmented patient with all 7 organs
        """
        # Extract real organ data
        real_organs = nhanes_patient['graph_features']
        demographics = nhanes_patient.get('demographics', {})
        
        # Generate lifestyle first (needed for liver/neural)
        lifestyle = self.generator.generate_lifestyle_conditioned(
            real_organs, demographics
        )
        
        # Generate liver (conditioned on metabolic + lifestyle)
        liver = self.generator.generate_liver_conditioned(
            real_organs, demographics, lifestyle
        )
        
        # Generate immune (conditioned on metabolic + liver)
        immune = self.generator.generate_immune_conditioned(
            real_organs, demographics, liver
        )
        
        # Generate neural (conditioned on cardiovascular + lifestyle)
        neural = self.generator.generate_neural_conditioned(
            real_organs, demographics, lifestyle
        )
        
        # Create augmented patient (preserve all original fields)
        augmented_patient = {
            'patient_id': nhanes_patient.get('patient_id'),
            'age': nhanes_patient.get('age'),
            'sex': nhanes_patient.get('sex'),
            'graph_features': {
                # Real organs (unchanged)
                'metabolic': real_organs.get('metabolic'),
                'cardiovascular': real_organs.get('cardiovascular'),
                'kidney': real_organs.get('kidney'),
                # Synthetic organs (generated)
                'liver': np.array([liver['ALT'], liver['AST']], dtype=np.float32),
                'immune': np.array([immune['WBC']], dtype=np.float32),
                'neural': np.array([neural['cognitive_score']], dtype=np.float32),
                'lifestyle': np.array([
                    lifestyle['alcohol_consumption'],
                    lifestyle['exercise_frequency'],
                    lifestyle['diet_quality'],
                    lifestyle['sleep_hours']
                ], dtype=np.float32)
            },
            'demographics': demographics,
            'disease_labels': nhanes_patient.get('disease_labels', {}),  # Preserve disease labels
            'has_complete_labels': nhanes_patient.get('has_complete_labels', False),
            'augmentation_metadata': {
                'liver_source': 'synthetic_conditioned',
                'immune_source': 'synthetic_conditioned',
                'neural_source': 'synthetic_conditioned',
                'lifestyle_source': 'synthetic_conditioned'
            }
        }
        
        return augmented_patient
    
    def augment_dataset(
        self,
        nhanes_path: str,
        output_path: str,
        apply_correlation_regularization: bool = True
    ) -> List[Dict]:
        """
        Augment entire NHANES dataset
        
        Args:
            nhanes_path: Path to original NHANES data
            output_path: Where to save augmented dataset
            apply_correlation_regularization: If True, adjust correlations to match targets
        
        Returns:
            List of augmented patients
        """
        print("="*80)
        print("AUGMENTING NHANES DATASET WITH SYNTHETIC MISSING ORGANS")
        print("="*80)
        
        # Load NHANES
        nhanes_patients = self.load_nhanes_data(nhanes_path)
        
        # Augment each patient
        print("\nGenerating synthetic organs for each patient...")
        augmented_patients = []
        
        for i, patient in enumerate(nhanes_patients):
            augmented = self.augment_patient_baseline(patient)
            augmented_patients.append(augmented)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(nhanes_patients)} patients")
        
        print(f"\n✓ Augmented {len(augmented_patients)} patients")
        
        # Apply correlation regularization
        if apply_correlation_regularization:
            print("\n" + "="*80)
            print("APPLYING CORRELATION REGULARIZATION")
            print("="*80)
            augmented_patients = self.apply_correlation_correction(augmented_patients)
        
        # Save
        augmented_data = {
            'patients': augmented_patients,
            'metadata': {
                'n_patients': len(augmented_patients),
                'augmentation_method': 'conditional_generation',
                'real_organs': ['metabolic', 'cardiovascular', 'kidney'],
                'synthetic_organs': ['liver', 'immune', 'neural', 'lifestyle'],
                'note': 'Synthetic organs generated conditioned on real NHANES data'
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(augmented_data, f)
        
        print(f"\n✓ Saved augmented dataset to {output_path}")
        
        # Validation
        self.validate_augmentation(augmented_patients)
        
        return augmented_patients
    
    def apply_correlation_correction(self, augmented_patients: List[Dict]) -> List[Dict]:
        """
        Apply correlation regularization to enforce target cross-organ correlations
        
        Args:
            augmented_patients: Patients with initial synthetic organs
        
        Returns:
            Patients with correlation-corrected synthetic organs
        """
        print("\nExtracting organ values for batch correction...")
        
        # Extract all values into arrays
        n_patients = len(augmented_patients)
        
        # Real organs
        glucoses = []
        triglycerides_list = []
        systolic_bps = []
        bmis = []
        
        # Synthetic organs
        alts = []
        wbcs = []
        cognitives = []
        
        for patient in augmented_patients:
            features = patient['graph_features']
            demographics = patient.get('demographics', {})
            
            # Only include if patient has valid real organ data
            metabolic = features.get('metabolic')
            cardiovascular = features.get('cardiovascular')
            
            # Skip if no real organs (shouldn't happen but be safe)
            if metabolic is None and cardiovascular is None:
                continue
            
            # Real organs (if available)
            if metabolic is not None and len(metabolic) > 0:
                glucoses.append(float(metabolic[0]))  # glucose
                if len(metabolic) > 3:
                    triglycerides_list.append(float(metabolic[3]))  # triglycerides
                else:
                    triglycerides_list.append(np.nan)
            else:
                glucoses.append(np.nan)
                triglycerides_list.append(np.nan)
            
            if cardiovascular is not None and len(cardiovascular) > 0:
                systolic_bps.append(float(cardiovascular[0]))  # systolic BP
            else:
                systolic_bps.append(np.nan)
            
            bmis.append(float(demographics.get('bmi', 27)))
            
            # Synthetic organs
            alts.append(float(features['liver'][0]))  # ALT
            wbcs.append(float(features['immune'][0]))  # WBC
            cognitives.append(float(features['neural'][0]))  # cognitive
        
        # Convert to arrays
        real_organs = {}
        if len(glucoses) > 0:
            real_organs['glucose'] = np.array(glucoses)
        if len(triglycerides_list) > 0:
            real_organs['triglycerides'] = np.array(triglycerides_list)
        if len(systolic_bps) > 0:
            real_organs['systolic_bp'] = np.array(systolic_bps)
        
        demographics_arrays = {
            'bmi': np.array(bmis)
        }
        
        synthetic_organs = {
            'ALT': np.array(alts),
            'WBC': np.array(wbcs),
            'cognitive': np.array(cognitives)
        }
        
        print(f"  Extracted {n_patients} patient organ values")
        
        # Debug: Check extracted values
        print(f"\nExtracted array statistics:")
        print(f"  Glucose: {len(real_organs.get('glucose', []))} values, {np.sum(~np.isnan(real_organs.get('glucose', [np.nan])))} valid")
        print(f"  ALT: {len(synthetic_organs.get('ALT', []))} values, {np.sum(~np.isnan(synthetic_organs.get('ALT', [np.nan])))} valid")
        print(f"  WBC: {len(synthetic_organs.get('WBC', []))} values, {np.sum(~np.isnan(synthetic_organs.get('WBC', [np.nan])))} valid")
        
        # Apply correlation regularization
        print("\nAdjusting correlations to match targets...")
        regularizer = CorrelationRegularizer(self.target_correlations)
        
        adjusted_synthetic = regularizer.adjust_batch_correlations(
            real_organs, synthetic_organs, demographics_arrays
        )
        
        # Debug: Check adjusted values
        print(f"\nAdjusted array statistics:")
        print(f"  ALT: {len(adjusted_synthetic.get('ALT', []))} values, {np.sum(~np.isnan(adjusted_synthetic.get('ALT', [np.nan])))} valid")
        print(f"  WBC: {len(adjusted_synthetic.get('WBC', []))} values, {np.sum(~np.isnan(adjusted_synthetic.get('WBC', [np.nan])))} valid")
        print(f"  Cognitive: {len(adjusted_synthetic.get('cognitive', []))} values, {np.sum(~np.isnan(adjusted_synthetic.get('cognitive', [np.nan])))} valid")
        if len(adjusted_synthetic.get('ALT', [])) > 0:
            print(f"  ALT sample values: {adjusted_synthetic['ALT'][:5]}")
        
        # Update patients with adjusted values
        print("\nUpdating patient records with adjusted values...")
        
        # Track which patients were included (match extraction loop)
        update_idx = 0
        for patient in augmented_patients:
            features = patient['graph_features']
            
            # Skip if no real organs (same condition as extraction)
            metabolic = features.get('metabolic')
            cardiovascular = features.get('cardiovascular')
            if metabolic is None and cardiovascular is None:
                continue
            
            # Update synthetic organs with adjusted values
            features['liver'][0] = adjusted_synthetic['ALT'][update_idx]
            features['immune'][0] = adjusted_synthetic['WBC'][update_idx]
            features['neural'][0] = adjusted_synthetic['cognitive'][update_idx]
            
            update_idx += 1
        
        print(f"  ✓ Correlation correction applied to {update_idx} patients")
        
        # Validate
        print("\nValidating adjusted correlations...")
        validation_results = regularizer.validate_correlations(
            real_organs, adjusted_synthetic, demographics_arrays
        )
        
        print("\nCorrelation Validation:")
        for corr_name, (actual, target, diff) in validation_results.items():
            status = "✓" if abs(diff) < 0.05 else "⚠"
            print(f"  {status} {corr_name}: actual={actual:.3f}, target={target:.3f}, diff={diff:+.3f}")
        
        return augmented_patients
    
    def validate_augmentation(self, augmented_patients: List[Dict]):
        """Validate synthetic organ statistics and correlations"""
        print("\n" + "="*80)
        print("VALIDATION: Synthetic Organ Statistics")
        print("="*80)
        
        # Extract synthetic organs
        alts = []
        wbcs = []
        cognitives = []
        glucoses = []
        bps = []
        
        for patient in augmented_patients[:10000]:  # Sample for speed
            features = patient['graph_features']
            alts.append(features['liver'][0])
            wbcs.append(features['immune'][0])
            cognitives.append(features['neural'][0])
            
            if features['metabolic'] is not None and len(features['metabolic']) > 0:
                glucoses.append(features['metabolic'][0])
            if features['cardiovascular'] is not None and len(features['cardiovascular']) > 0:
                bps.append(features['cardiovascular'][0])
        
        # Convert to arrays and use nanmean/nanstd
        alts = np.array(alts)
        wbcs = np.array(wbcs)
        cognitives = np.array(cognitives)
        glucoses = np.array(glucoses) if glucoses else np.array([])
        bps = np.array(bps) if bps else np.array([])
        
        # Statistics (using nanmean/nanstd to handle any NaN)
        print(f"\nALT: mean={np.nanmean(alts):.1f}, std={np.nanstd(alts):.1f}")
        print(f"  Expected: mean=25, std=10")
        
        print(f"\nWBC: mean={np.nanmean(wbcs):.2f}, std={np.nanstd(wbcs):.2f}")
        print(f"  Expected: mean=7.0, std=2.0")
        
        print(f"\nCognitive: mean={np.nanmean(cognitives):.2f}, std={np.nanstd(cognitives):.2f}")
        print(f"  Expected: mean=0.85, std=0.12")
        
        # Correlations (remove NaN pairs)
        if len(glucoses) > 100 and len(alts) > 100:
            min_len = min(len(glucoses), len(alts))
            g = glucoses[:min_len]
            a = alts[:min_len]
            valid = ~(np.isnan(g) | np.isnan(a))
            if np.sum(valid) > 100:
                corr_glucose_alt = np.corrcoef(g[valid], a[valid])[0, 1]
                print(f"\nGlucose-ALT correlation: {corr_glucose_alt:.3f}")
                print(f"  Expected: ~0.25")
        
        if len(bps) > 100 and len(cognitives) > 100:
            min_len = min(len(bps), len(cognitives))
            b = bps[:min_len]
            c = cognitives[:min_len]
            valid = ~(np.isnan(b) | np.isnan(c))
            if np.sum(valid) > 100:
                corr_bp_cog = np.corrcoef(b[valid], c[valid])[0, 1]
                print(f"\nBP-Cognitive correlation: {corr_bp_cog:.3f}")
                print(f"  Expected: ~-0.20")


def main():
    """Augment NHANES dataset with synthetic missing organs"""
    
    augmenter = NHANESAugmenter()
    
    # Augment dataset
    augmented_patients = augmenter.augment_dataset(
        nhanes_path='./data/nhanes_all_135310.pkl',
        output_path='./data/nhanes_augmented_complete.pkl'
    )
    
    print("\n" + "="*80)
    print("✓ AUGMENTATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Train GNN-Transformer on augmented dataset:")
    print("   python3 train_two_stage.py \\")
    print("     --pretrain_data ./data/nhanes_augmented_complete.pkl \\")
    print("     --finetune_data ./data/nhanes_augmented_complete.pkl")
    print("\n2. Model will learn cross-organ coupling from:")
    print("   - Real metabolic/CV/kidney dynamics")
    print("   - Synthetic liver/immune/neural/lifestyle (conditioned on real data)")
    print("\n3. Cross-organ correlations preserved:")
    print("   - Glucose ↔ ALT (metabolic-liver)")
    print("   - BP ↔ Cognitive (cardiovascular-neural)")
    print("   - BMI ↔ WBC (inflammation)")


if __name__ == '__main__':
    main()
