#!/usr/bin/env python3
"""
Domain Knowledge Rules for Organ Dynamics

When training data is insufficient, use medical domain knowledge
to ensure realistic organ behavior.
"""

import torch
from typing import Dict


class DomainKnowledgeRules:
    """
    Medical domain knowledge for organ dynamics
    
    Used to supplement learned dynamics when training data is limited
    """
    
    @staticmethod
    def apply_liver_dynamics(
        current_liver: torch.Tensor,
        lifestyle: Dict[str, float],
        time_delta_months: int = 1
    ) -> torch.Tensor:
        """
        Apply domain knowledge for liver changes
        
        Medical facts:
        - Alcohol consumption increases ALT/AST
        - Poor diet + sedentary lifestyle → fatty liver → elevated ALT
        - Exercise + good diet can reduce liver enzymes
        
        Args:
            current_liver: [ALT, AST] in U/L
            lifestyle: Dict with exercise, alcohol, diet
            time_delta_months: Number of months
        
        Returns:
            delta_liver: Change in [ALT, AST]
        """
        ALT, AST = current_liver[0].item(), current_liver[1].item()
        
        # Alcohol effect on liver enzymes
        alcohol = lifestyle.get('alcohol_consumption', 0.5)  # 0-1 scale
        
        # Heavy drinking (>0.7) increases ALT by ~2-5 U/L per month
        # Moderate (0.3-0.7) increases by ~0.5-2 U/L per month
        # Light (<0.3) minimal effect
        if alcohol > 0.7:
            alcohol_effect_alt = (alcohol - 0.7) * 15 * time_delta_months  # Up to 4.5 U/L/month
            alcohol_effect_ast = (alcohol - 0.7) * 10 * time_delta_months
        elif alcohol > 0.3:
            alcohol_effect_alt = (alcohol - 0.3) * 5 * time_delta_months
            alcohol_effect_ast = (alcohol - 0.3) * 3 * time_delta_months
        else:
            alcohol_effect_alt = 0
            alcohol_effect_ast = 0
        
        # Diet and exercise effect
        diet_quality = lifestyle.get('diet_quality', 0.5)
        exercise = lifestyle.get('exercise_frequency', 0.5)
        
        # Poor diet + no exercise → metabolic stress on liver
        metabolic_stress = (1 - diet_quality) * (1 - exercise)
        metabolic_effect_alt = metabolic_stress * 3 * time_delta_months  # Up to 3 U/L/month
        
        # Good diet + exercise can reduce liver enzymes (if elevated)
        if ALT > 40:  # If elevated
            improvement_factor = diet_quality * exercise
            improvement_effect_alt = -improvement_factor * 2 * time_delta_months
        else:
            improvement_effect_alt = 0
        
        # Total change
        delta_alt = alcohol_effect_alt + metabolic_effect_alt + improvement_effect_alt
        delta_ast = alcohol_effect_ast + metabolic_effect_alt * 0.5 + improvement_effect_alt * 0.5
        
        # Physiological limits
        # ALT rarely goes above 200 without severe disease
        # ALT rarely goes below 10
        if ALT + delta_alt > 200:
            delta_alt = 200 - ALT
        if ALT + delta_alt < 10:
            delta_alt = 10 - ALT
        
        if AST + delta_ast > 150:
            delta_ast = 150 - AST
        if AST + delta_ast < 10:
            delta_ast = 10 - AST
        
        return torch.tensor([delta_alt, delta_ast], dtype=torch.float32)
    
    @staticmethod
    def apply_metabolic_dynamics(
        current_metabolic: torch.Tensor,
        lifestyle: Dict[str, float],
        time_delta_months: int = 1
    ) -> torch.Tensor:
        """
        Apply domain knowledge for metabolic changes
        
        Medical facts:
        - Poor diet + sedentary → insulin resistance → elevated glucose
        - Exercise improves insulin sensitivity
        - Weight loss improves metabolic health
        """
        # [glucose, HbA1c, insulin, triglycerides]
        glucose = current_metabolic[0].item()
        
        diet_quality = lifestyle.get('diet_quality', 0.5)
        exercise = lifestyle.get('exercise_frequency', 0.5)
        
        # Poor diet effect on glucose
        diet_effect = (1 - diet_quality) * 2 * time_delta_months  # Up to 2 mg/dL/month
        
        # Exercise effect (protective)
        if glucose > 100:  # If elevated
            exercise_effect = -exercise * 1.5 * time_delta_months
        else:
            exercise_effect = 0
        
        delta_glucose = diet_effect + exercise_effect
        
        # HbA1c follows glucose (slower)
        delta_hba1c = delta_glucose * 0.01  # Rough conversion
        
        # Insulin resistance
        delta_insulin = (1 - diet_quality) * (1 - exercise) * 0.5 * time_delta_months
        
        # Triglycerides
        delta_triglycerides = (1 - diet_quality) * 5 * time_delta_months - exercise * 3 * time_delta_months
        
        # Limits
        if glucose + delta_glucose > 300:
            delta_glucose = 300 - glucose
        if glucose + delta_glucose < 60:
            delta_glucose = 60 - glucose
        
        return torch.tensor([
            delta_glucose,
            delta_hba1c,
            delta_insulin,
            delta_triglycerides
        ], dtype=torch.float32)
    
    @staticmethod
    def apply_cardiovascular_dynamics(
        current_cv: torch.Tensor,
        lifestyle: Dict[str, float],
        time_delta_months: int = 1
    ) -> torch.Tensor:
        """
        Apply domain knowledge for cardiovascular changes
        
        Medical facts:
        - Alcohol increases blood pressure
        - Exercise lowers blood pressure
        - Diet affects cholesterol
        """
        # [systolic_bp, diastolic_bp, total_chol, HDL, LDL]
        systolic_bp = current_cv[0].item()
        
        alcohol = lifestyle.get('alcohol_consumption', 0.5)
        exercise = lifestyle.get('exercise_frequency', 0.5)
        
        # Alcohol effect on BP
        alcohol_effect_bp = alcohol * 1.5 * time_delta_months  # Up to 1.5 mmHg/month
        
        # Exercise effect (protective)
        if systolic_bp > 120:
            exercise_effect_bp = -exercise * 1 * time_delta_months
        else:
            exercise_effect_bp = 0
        
        delta_systolic = alcohol_effect_bp + exercise_effect_bp
        delta_diastolic = delta_systolic * 0.6  # Diastolic changes less
        
        # Cholesterol (simplified)
        diet_quality = lifestyle.get('diet_quality', 0.5)
        delta_chol = (1 - diet_quality) * 2 * time_delta_months
        delta_hdl = exercise * 0.5 * time_delta_months  # Exercise increases HDL
        delta_ldl = (1 - diet_quality) * 1.5 * time_delta_months
        
        # Limits
        if systolic_bp + delta_systolic > 200:
            delta_systolic = 200 - systolic_bp
        if systolic_bp + delta_systolic < 90:
            delta_systolic = 90 - systolic_bp
        
        return torch.tensor([
            delta_systolic,
            delta_diastolic,
            delta_chol,
            delta_hdl,
            delta_ldl
        ], dtype=torch.float32)
    
    @staticmethod
    def apply_all_rules(
        current_organs: Dict[str, torch.Tensor],
        lifestyle: Dict[str, float],
        time_delta_months: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Apply domain knowledge rules to all organs
        
        Returns:
            deltas: Predicted changes for each organ
        """
        deltas = {}
        
        # Liver (use domain rules - training data is bad)
        if 'liver' in current_organs:
            deltas['liver'] = DomainKnowledgeRules.apply_liver_dynamics(
                current_organs['liver'],
                lifestyle,
                time_delta_months
            )
        
        # Metabolic (supplement learned dynamics)
        if 'metabolic' in current_organs:
            deltas['metabolic'] = DomainKnowledgeRules.apply_metabolic_dynamics(
                current_organs['metabolic'],
                lifestyle,
                time_delta_months
            )
        
        # Cardiovascular (supplement learned dynamics)
        if 'cardiovascular' in current_organs:
            deltas['cardiovascular'] = DomainKnowledgeRules.apply_cardiovascular_dynamics(
                current_organs['cardiovascular'],
                lifestyle,
                time_delta_months
            )
        
        # Other organs - use zero change for now (can add rules later)
        for organ in ['kidney', 'immune', 'neural', 'lifestyle']:
            if organ in current_organs:
                deltas[organ] = torch.zeros_like(current_organs[organ])
        
        return deltas


def test_domain_rules():
    """Test domain knowledge rules"""
    
    # Test case: Heavy drinker with poor lifestyle
    current_liver = torch.tensor([45.0, 38.0])  # Mildly elevated ALT
    lifestyle_bad = {
        'alcohol_consumption': 0.9,  # Heavy drinking
        'diet_quality': 0.3,  # Poor diet
        'exercise_frequency': 0.2  # Sedentary
    }
    
    delta = DomainKnowledgeRules.apply_liver_dynamics(current_liver, lifestyle_bad, time_delta_months=1)
    print(f"Heavy drinker, poor lifestyle:")
    print(f"  Current: ALT={current_liver[0]:.1f}, AST={current_liver[1]:.1f}")
    print(f"  Delta (1 month): ALT +{delta[0]:.1f}, AST +{delta[1]:.1f}")
    print(f"  After 1 month: ALT={current_liver[0]+delta[0]:.1f}, AST={current_liver[1]+delta[1]:.1f}")
    
    # Test case: Healthy lifestyle
    lifestyle_good = {
        'alcohol_consumption': 0.1,  # Minimal drinking
        'diet_quality': 0.8,  # Good diet
        'exercise_frequency': 0.8  # Active
    }
    
    delta_good = DomainKnowledgeRules.apply_liver_dynamics(current_liver, lifestyle_good, time_delta_months=1)
    print(f"\nHealthy lifestyle:")
    print(f"  Current: ALT={current_liver[0]:.1f}, AST={current_liver[1]:.1f}")
    print(f"  Delta (1 month): ALT {delta_good[0]:+.1f}, AST {delta_good[1]:+.1f}")
    print(f"  After 1 month: ALT={current_liver[0]+delta_good[0]:.1f}, AST={current_liver[1]+delta_good[1]:.1f}")


if __name__ == '__main__':
    test_domain_rules()
