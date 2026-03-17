#!/usr/bin/env python3
"""
Advanced research analysis tools for population health studies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PopulationHealthAnalyzer:
    def __init__(self, data_path: str = "data/synthetic/complete_patient_data.csv"):
        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df):,} patients")
    
    def analyze_comorbidity_patterns(self):
        """Analyze disease comorbidity patterns"""
        logger.info("\n" + "="*80)
        logger.info("COMORBIDITY ANALYSIS")
        logger.info("="*80)
        
        diseases = ['hypertension', 'diabetes', 'heart_disease', 'stroke', 
                   'cancer', 'copd', 'depression', 'anxiety']
        
        # Calculate comorbidity matrix
        comorbidity_matrix = pd.DataFrame(index=diseases, columns=diseases, dtype=float)
        
        for disease1 in diseases:
            for disease2 in diseases:
                if disease1 == disease2:
                    comorbidity_matrix.loc[disease1, disease2] = self.df[disease1].mean()
                else:
                    # Conditional probability: P(disease2 | disease1)
                    has_disease1 = self.df[disease1] == True
                    if has_disease1.sum() > 0:
                        prob = self.df[has_disease1][disease2].mean()
                        comorbidity_matrix.loc[disease1, disease2] = prob
                    else:
                        comorbidity_matrix.loc[disease1, disease2] = 0
        
        logger.info("\nComorbidity Matrix (Conditional Probabilities):")
        logger.info("\nP(Column Disease | Row Disease)")
        logger.info(comorbidity_matrix.to_string())
        
        # Find strongest comorbidities
        logger.info("\n\nStrongest Comorbidities:")
        for disease1 in diseases:
            for disease2 in diseases:
                if disease1 != disease2:
                    prob = comorbidity_matrix.loc[disease1, disease2]
                    if prob > 0.3:  # Show if >30% conditional probability
                        logger.info(f"  {disease1} → {disease2}: {prob:.1%}")
        
        return comorbidity_matrix
    
    def analyze_risk_factors_by_disease(self, disease: str):
        """Analyze risk factor distributions for a specific disease"""
        logger.info(f"\n" + "="*80)
        logger.info(f"RISK FACTOR ANALYSIS: {disease.upper()}")
        logger.info("="*80)
        
        has_disease = self.df[disease] == True
        no_disease = self.df[disease] == False
        
        risk_factors = {
            'Age': 'age',
            'BMI': 'bmi',
            'Systolic BP': 'systolic_bp',
            'Glucose': 'glucose_mmol_l',
            'HbA1c': 'hba1c_percent',
            'Total Cholesterol': 'total_cholesterol_mmol_l',
            'Exercise (hrs/week)': 'exercise_hours_per_week',
            'Sleep (hrs/night)': 'sleep_hours_per_night',
            'Alcohol (units/week)': 'alcohol_units_per_week'
        }
        
        logger.info(f"\nPatients with {disease}: {has_disease.sum():,} ({has_disease.mean():.1%})")
        logger.info(f"Patients without {disease}: {no_disease.sum():,} ({no_disease.mean():.1%})")
        
        logger.info(f"\n{'Risk Factor':<25} {'With Disease':<20} {'Without Disease':<20} {'Difference'}")
        logger.info("-" * 85)
        
        for name, col in risk_factors.items():
            with_mean = self.df[has_disease][col].mean()
            without_mean = self.df[no_disease][col].mean()
            diff = with_mean - without_mean
            
            logger.info(f"{name:<25} {with_mean:>8.2f} ± {self.df[has_disease][col].std():>5.2f}   "
                       f"{without_mean:>8.2f} ± {self.df[no_disease][col].std():>5.2f}   "
                       f"{diff:>+7.2f}")
        
        # Categorical risk factors
        logger.info("\n\nCategorical Risk Factors:")
        
        # Smoking
        logger.info("\nSmoking Status:")
        for status in ['never', 'former', 'current']:
            with_pct = (self.df[has_disease]['smoking_status'] == status).mean()
            without_pct = (self.df[no_disease]['smoking_status'] == status).mean()
            logger.info(f"  {status:10s}: With={with_pct:.1%}, Without={without_pct:.1%}, "
                       f"Diff={with_pct-without_pct:+.1%}")
        
        # Gender
        logger.info("\nGender:")
        for gender in ['male', 'female']:
            with_pct = (self.df[has_disease]['gender'] == gender).mean()
            without_pct = (self.df[no_disease]['gender'] == gender).mean()
            logger.info(f"  {gender:10s}: With={with_pct:.1%}, Without={without_pct:.1%}, "
                       f"Diff={with_pct-without_pct:+.1%}")
    
    def analyze_age_stratified_prevalence(self):
        """Analyze disease prevalence by age groups"""
        logger.info("\n" + "="*80)
        logger.info("AGE-STRATIFIED DISEASE PREVALENCE")
        logger.info("="*80)
        
        # Define age groups
        self.df['age_group'] = pd.cut(self.df['age'], 
                                      bins=[0, 30, 40, 50, 60, 70, 120],
                                      labels=['<30', '30-39', '40-49', '50-59', '60-69', '70+'])
        
        diseases = ['hypertension', 'diabetes', 'heart_disease', 'cancer']
        
        logger.info("\nPrevalence by Age Group:")
        logger.info(f"\n{'Age Group':<12} {'Hypertension':<15} {'Diabetes':<15} {'Heart Disease':<15} {'Cancer':<15}")
        logger.info("-" * 72)
        
        for age_group in ['<30', '30-39', '40-49', '50-59', '60-69', '70+']:
            group_data = self.df[self.df['age_group'] == age_group]
            if len(group_data) > 0:
                row = f"{age_group:<12}"
                for disease in diseases:
                    prev = group_data[disease].mean()
                    row += f" {prev:>6.1%}         "
                logger.info(row)
    
    def identify_high_risk_subpopulations(self):
        """Identify high-risk patient subpopulations"""
        logger.info("\n" + "="*80)
        logger.info("HIGH-RISK SUBPOPULATION IDENTIFICATION")
        logger.info("="*80)
        
        # Define high-risk criteria
        high_risk_groups = []
        
        # Group 1: Metabolic syndrome
        metabolic_syndrome = (
            (self.df['bmi'] > 30) &
            (self.df['systolic_bp'] > 130) &
            (self.df['glucose_mmol_l'] > 5.6)
        )
        high_risk_groups.append(('Metabolic Syndrome', metabolic_syndrome))
        
        # Group 2: Cardiovascular high risk
        cvd_high_risk = (
            (self.df['age'] > 60) &
            (self.df['systolic_bp'] > 140) &
            (self.df['total_cholesterol_mmol_l'] > 6.0)
        )
        high_risk_groups.append(('CVD High Risk', cvd_high_risk))
        
        # Group 3: Diabetes high risk
        diabetes_high_risk = (
            (self.df['bmi'] > 30) &
            (self.df['hba1c_percent'] > 5.7) &
            (self.df['exercise_hours_per_week'] < 2)
        )
        high_risk_groups.append(('Diabetes High Risk', diabetes_high_risk))
        
        # Group 4: Multiple risk factors
        multiple_risks = (
            (self.df['smoking_status'] == 'current') &
            (self.df['bmi'] > 30) &
            (self.df['exercise_hours_per_week'] < 2) &
            (self.df['alcohol_units_per_week'] > 14)
        )
        high_risk_groups.append(('Multiple Lifestyle Risks', multiple_risks))
        
        logger.info("\nHigh-Risk Subpopulations:")
        for group_name, group_mask in high_risk_groups:
            count = group_mask.sum()
            pct = group_mask.mean()
            logger.info(f"\n{group_name}:")
            logger.info(f"  Population: {count:,} ({pct:.1%})")
            
            # Disease prevalence in this group
            group_data = self.df[group_mask]
            if len(group_data) > 0:
                logger.info(f"  Hypertension: {group_data['hypertension'].mean():.1%}")
                logger.info(f"  Diabetes: {group_data['diabetes'].mean():.1%}")
                logger.info(f"  Heart Disease: {group_data['heart_disease'].mean():.1%}")
    
    def generate_summary_report(self, output_file: str = "research_analysis_report.txt"):
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE RESEARCH REPORT")
        logger.info("="*80)
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("POPULATION HEALTH RESEARCH ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {len(self.df):,} patients\n\n")
            
            # Demographics
            f.write("DEMOGRAPHICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Age: {self.df['age'].mean():.1f} ± {self.df['age'].std():.1f} years\n")
            f.write(f"Gender: {(self.df['gender']=='male').mean():.1%} male, "
                   f"{(self.df['gender']=='female').mean():.1%} female\n")
            f.write(f"BMI: {self.df['bmi'].mean():.1f} ± {self.df['bmi'].std():.1f}\n\n")
            
            # Disease prevalence
            f.write("DISEASE PREVALENCE\n")
            f.write("-" * 80 + "\n")
            diseases = ['hypertension', 'diabetes', 'heart_disease', 'stroke', 
                       'cancer', 'copd', 'depression', 'anxiety']
            for disease in diseases:
                prev = self.df[disease].mean()
                count = self.df[disease].sum()
                f.write(f"{disease.replace('_', ' ').title()}: {count:,} ({prev:.1%})\n")
        
        logger.info(f"\n✅ Report saved to {output_file}")


if __name__ == "__main__":
    analyzer = PopulationHealthAnalyzer()
    
    # Run all analyses
    analyzer.analyze_comorbidity_patterns()
    analyzer.analyze_risk_factors_by_disease('diabetes')
    analyzer.analyze_risk_factors_by_disease('heart_disease')
    analyzer.analyze_age_stratified_prevalence()
    analyzer.identify_high_risk_subpopulations()
    analyzer.generate_summary_report()
    
    logger.info("\n" + "="*80)
    logger.info("✅ Research analysis complete!")
    logger.info("="*80)
