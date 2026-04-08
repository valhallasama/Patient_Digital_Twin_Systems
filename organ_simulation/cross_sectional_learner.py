#!/usr/bin/env python3
"""
Cross-Sectional Pattern Learning

For organs with constant temporal data, learn patterns from cross-sectional
correlations in the 135K patient dataset.

Example: "Patients with high alcohol consumption have ALT values 30% higher
than patients with low alcohol consumption"
"""

import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy import stats
from collections import defaultdict


class CrossSectionalPatternLearner:
    """
    Learn organ-lifestyle correlations from cross-sectional data
    
    When temporal data is unavailable, use population patterns to infer
    how lifestyle affects organ biomarkers.
    """
    
    def __init__(self, data_path: str = './data/nhanes_all_135310.pkl'):
        self.data_path = data_path
        self.patterns = {}
        self.correlations = {}
        
    def extract_patterns(self):
        """
        Extract cross-sectional patterns from NHANES data
        
        For each organ, find correlations with:
        - Age
        - BMI
        - Lifestyle factors (if available)
        - Other organ states
        """
        print("Loading NHANES data...")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        patients = data['patients']
        print(f"Loaded {len(patients):,} patients")
        
        # Extract patterns for each organ
        self._extract_metabolic_patterns(patients)
        self._extract_cardiovascular_patterns(patients)
        self._extract_kidney_patterns(patients)
        
        # For constant organs, extract population statistics
        self._extract_liver_population_stats(patients)
        self._extract_immune_population_stats(patients)
        
        return self.patterns
    
    def _extract_metabolic_patterns(self, patients: List[Dict]):
        """Extract metabolic-age-BMI correlations"""
        print("\nExtracting metabolic patterns...")
        
        ages = []
        glucose_values = []
        hba1c_values = []
        
        for p in patients[:10000]:  # Sample for speed
            if 'age' in p and 'metabolic' in p['graph_features']:
                ages.append(p['age'])
                metabolic = p['graph_features']['metabolic']
                glucose_values.append(metabolic[0])
                hba1c_values.append(metabolic[1])
        
        # Calculate age correlation
        if len(ages) > 100:
            glucose_age_corr, _ = stats.pearsonr(ages, glucose_values)
            hba1c_age_corr, _ = stats.pearsonr(ages, hba1c_values)
            
            self.patterns['metabolic'] = {
                'glucose_age_slope': np.polyfit(ages, glucose_values, 1)[0],
                'hba1c_age_slope': np.polyfit(ages, hba1c_values, 1)[0],
                'glucose_age_correlation': glucose_age_corr,
                'hba1c_age_correlation': hba1c_age_corr
            }
            
            print(f"  Glucose-age correlation: {glucose_age_corr:.3f}")
            print(f"  Glucose increases {self.patterns['metabolic']['glucose_age_slope']:.2f} mg/dL per year")
            print(f"  HbA1c-age correlation: {hba1c_age_corr:.3f}")
    
    def _extract_cardiovascular_patterns(self, patients: List[Dict]):
        """Extract cardiovascular-age correlations"""
        print("\nExtracting cardiovascular patterns...")
        
        ages = []
        systolic_values = []
        
        for p in patients[:10000]:
            if 'age' in p and 'cardiovascular' in p['graph_features']:
                ages.append(p['age'])
                cv = p['graph_features']['cardiovascular']
                systolic_values.append(cv[0])
        
        if len(ages) > 100:
            bp_age_corr, _ = stats.pearsonr(ages, systolic_values)
            
            self.patterns['cardiovascular'] = {
                'systolic_age_slope': np.polyfit(ages, systolic_values, 1)[0],
                'systolic_age_correlation': bp_age_corr
            }
            
            print(f"  Systolic BP-age correlation: {bp_age_corr:.3f}")
            print(f"  BP increases {self.patterns['cardiovascular']['systolic_age_slope']:.2f} mmHg per year")
    
    def _extract_kidney_patterns(self, patients: List[Dict]):
        """Extract kidney-age correlations"""
        print("\nExtracting kidney patterns...")
        
        ages = []
        creatinine_values = []
        
        for p in patients[:10000]:
            if 'age' in p and 'kidney' in p['graph_features']:
                ages.append(p['age'])
                kidney = p['graph_features']['kidney']
                creatinine_values.append(kidney[0])
        
        if len(ages) > 100:
            creat_age_corr, _ = stats.pearsonr(ages, creatinine_values)
            
            self.patterns['kidney'] = {
                'creatinine_age_slope': np.polyfit(ages, creatinine_values, 1)[0],
                'creatinine_age_correlation': creat_age_corr
            }
            
            print(f"  Creatinine-age correlation: {creat_age_corr:.3f}")
            print(f"  Creatinine changes {self.patterns['kidney']['creatinine_age_slope']:.4f} mg/dL per year")
    
    def _extract_liver_population_stats(self, patients: List[Dict]):
        """
        For liver (constant data), extract population statistics
        
        Since all values are constant, we can't learn correlations.
        Instead, document the population baseline.
        """
        print("\nExtracting liver population statistics...")
        
        liver_values = []
        for p in patients[:1000]:
            if 'liver' in p['graph_features']:
                liver_values.append(p['graph_features']['liver'])
        
        if liver_values:
            liver_array = np.array(liver_values)
            self.patterns['liver'] = {
                'population_mean_alt': np.mean(liver_array[:, 0]),
                'population_mean_ast': np.mean(liver_array[:, 1]),
                'note': 'Data is constant - using domain knowledge instead'
            }
            print(f"  Population ALT: {self.patterns['liver']['population_mean_alt']:.1f} U/L (constant)")
            print(f"  Population AST: {self.patterns['liver']['population_mean_ast']:.1f} U/L (constant)")
            print(f"  ⚠️  Cannot learn correlations from constant data")
    
    def _extract_immune_population_stats(self, patients: List[Dict]):
        """Extract immune population statistics"""
        print("\nExtracting immune population statistics...")
        
        immune_values = []
        for p in patients[:1000]:
            if 'immune' in p['graph_features']:
                immune_values.append(p['graph_features']['immune'][0])
        
        if immune_values:
            self.patterns['immune'] = {
                'population_mean_wbc': np.mean(immune_values),
                'note': 'Data is constant - using domain knowledge instead'
            }
            print(f"  Population WBC: {self.patterns['immune']['population_mean_wbc']:.1f} K/μL (constant)")
    
    def predict_age_effect(
        self,
        organ: str,
        current_value: float,
        age_years: float,
        time_delta_months: int = 1
    ) -> float:
        """
        Predict organ change based on age progression
        
        Uses learned age-correlation slopes from cross-sectional data
        """
        if organ not in self.patterns:
            return 0.0
        
        patterns = self.patterns[organ]
        
        # Age effect (per month)
        time_delta_years = time_delta_months / 12.0
        
        if organ == 'metabolic':
            # Glucose increases with age
            delta = patterns.get('glucose_age_slope', 0) * time_delta_years
        elif organ == 'cardiovascular':
            # BP increases with age
            delta = patterns.get('systolic_age_slope', 0) * time_delta_years
        elif organ == 'kidney':
            # Creatinine increases with age
            delta = patterns.get('creatinine_age_slope', 0) * time_delta_years
        else:
            delta = 0.0
        
        return delta
    
    def save_patterns(self, path: str = './models/cross_sectional_patterns.pkl'):
        """Save learned patterns"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.patterns, f)
        print(f"\n✓ Saved cross-sectional patterns to {path}")
    
    def load_patterns(self, path: str = './models/cross_sectional_patterns.pkl'):
        """Load learned patterns"""
        import pickle
        try:
            with open(path, 'rb') as f:
                self.patterns = pickle.load(f)
            print(f"✓ Loaded cross-sectional patterns from {path}")
            return True
        except FileNotFoundError:
            print(f"⚠️  Pattern file not found: {path}")
            return False


def main():
    """Extract and save cross-sectional patterns"""
    print("="*80)
    print("CROSS-SECTIONAL PATTERN EXTRACTION")
    print("="*80)
    
    learner = CrossSectionalPatternLearner()
    patterns = learner.extract_patterns()
    
    print("\n" + "="*80)
    print("LEARNED PATTERNS SUMMARY")
    print("="*80)
    
    for organ, pattern_dict in patterns.items():
        print(f"\n{organ.upper()}:")
        for key, value in pattern_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    learner.save_patterns()
    
    print("\n✓ Cross-sectional pattern extraction complete!")


if __name__ == '__main__':
    main()
