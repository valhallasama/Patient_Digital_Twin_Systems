"""
Intervention Impact Calculator
Shows how lifestyle changes affect disease risk
"If you exercise 2h more per week, diabetes risk drops 10%"
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class InterventionImpact:
    """Impact of a single intervention"""
    intervention: str
    description: str
    risk_reduction: float  # Percentage reduction (0-1)
    time_to_effect: int  # Days until effect
    confidence: float  # Evidence confidence (0-1)
    source: str  # Medical literature reference


class InterventionCalculator:
    """
    Calculate impact of lifestyle interventions on disease risk
    Based on medical literature and clinical trials
    """
    
    def __init__(self):
        self.interventions = self._load_intervention_data()
    
    def _load_intervention_data(self) -> Dict[str, List[InterventionImpact]]:
        """
        Load intervention impacts from medical literature
        Data from meta-analyses and clinical trials
        """
        return {
            'diabetes': [
                InterventionImpact(
                    intervention='exercise_increase',
                    description='Increase exercise by 150 min/week (moderate intensity)',
                    risk_reduction=0.58,  # 58% reduction
                    time_to_effect=365,  # 1 year
                    confidence=0.95,
                    source='Diabetes Prevention Program, NEJM 2002;346:393-403'
                ),
                InterventionImpact(
                    intervention='weight_loss',
                    description='Lose 7% of body weight',
                    risk_reduction=0.58,  # 58% reduction
                    time_to_effect=365,
                    confidence=0.95,
                    source='DPP Research Group, Lancet 2009;374:1677-1686'
                ),
                InterventionImpact(
                    intervention='mediterranean_diet',
                    description='Switch to Mediterranean diet',
                    risk_reduction=0.30,  # 30% reduction
                    time_to_effect=180,
                    confidence=0.88,
                    source='PREDIMED Study, Diabetes Care 2011;34:14-19'
                ),
                InterventionImpact(
                    intervention='sleep_improvement',
                    description='Increase sleep to 7-8 hours/night',
                    risk_reduction=0.28,  # 28% reduction
                    time_to_effect=90,
                    confidence=0.82,
                    source='Cappuccio et al, Diabetologia 2010;53:2538-2545'
                ),
                InterventionImpact(
                    intervention='stress_reduction',
                    description='Stress management (meditation, yoga)',
                    risk_reduction=0.23,  # 23% reduction
                    time_to_effect=90,
                    confidence=0.75,
                    source='Rosmond et al, Metabolism 2000;49:1130-1134'
                ),
                InterventionImpact(
                    intervention='metformin',
                    description='Metformin 850mg twice daily',
                    risk_reduction=0.31,  # 31% reduction
                    time_to_effect=90,
                    confidence=0.92,
                    source='DPP Research Group, NEJM 2002;346:393-403'
                ),
            ],
            'cardiovascular': [
                InterventionImpact(
                    intervention='exercise_increase',
                    description='Increase exercise by 150 min/week',
                    risk_reduction=0.35,  # 35% reduction
                    time_to_effect=180,
                    confidence=0.90,
                    source='Nocon et al, Circulation 2008;117:2358-2368'
                ),
                InterventionImpact(
                    intervention='smoking_cessation',
                    description='Quit smoking',
                    risk_reduction=0.50,  # 50% reduction
                    time_to_effect=365,
                    confidence=0.95,
                    source='Critchley et al, Cochrane Database 2012'
                ),
                InterventionImpact(
                    intervention='bp_control',
                    description='Control BP to <130/80 mmHg',
                    risk_reduction=0.25,  # 25% reduction
                    time_to_effect=180,
                    confidence=0.92,
                    source='SPRINT Trial, NEJM 2015;373:2103-2116'
                ),
                InterventionImpact(
                    intervention='statin_therapy',
                    description='Statin therapy for high LDL',
                    risk_reduction=0.30,  # 30% reduction
                    time_to_effect=365,
                    confidence=0.95,
                    source='CTT Collaboration, Lancet 2010;376:1670-1681'
                ),
                InterventionImpact(
                    intervention='mediterranean_diet',
                    description='Switch to Mediterranean diet',
                    risk_reduction=0.30,  # 30% reduction
                    time_to_effect=365,
                    confidence=0.90,
                    source='PREDIMED Trial, NEJM 2013;368:1279-1290'
                ),
            ],
            'hypertension': [
                InterventionImpact(
                    intervention='sodium_reduction',
                    description='Reduce sodium to <2g/day',
                    risk_reduction=0.20,  # 20% reduction in BP
                    time_to_effect=30,
                    confidence=0.88,
                    source='He et al, BMJ 2013;346:f1325'
                ),
                InterventionImpact(
                    intervention='weight_loss',
                    description='Lose 5kg body weight',
                    risk_reduction=0.15,  # 15% reduction in BP
                    time_to_effect=90,
                    confidence=0.85,
                    source='Neter et al, Arch Intern Med 2003;163:1343-1350'
                ),
                InterventionImpact(
                    intervention='dash_diet',
                    description='DASH diet (fruits, vegetables, low-fat dairy)',
                    risk_reduction=0.11,  # 11 mmHg reduction
                    time_to_effect=60,
                    confidence=0.92,
                    source='Appel et al, NEJM 1997;336:1117-1124'
                ),
                InterventionImpact(
                    intervention='exercise_increase',
                    description='Aerobic exercise 150 min/week',
                    risk_reduction=0.10,  # 10% reduction in BP
                    time_to_effect=90,
                    confidence=0.88,
                    source='Cornelissen et al, Hypertension 2013;61:1360-1383'
                ),
            ],
            'chronic_kidney_disease': [
                InterventionImpact(
                    intervention='bp_control',
                    description='Control BP to <130/80 mmHg',
                    risk_reduction=0.40,  # 40% slower progression
                    time_to_effect=180,
                    confidence=0.90,
                    source='KDIGO Guidelines 2012'
                ),
                InterventionImpact(
                    intervention='glucose_control',
                    description='Control HbA1c to <7%',
                    risk_reduction=0.35,  # 35% slower progression
                    time_to_effect=365,
                    confidence=0.88,
                    source='DCCT/EDIC, NEJM 2011;365:2366-2376'
                ),
                InterventionImpact(
                    intervention='ace_inhibitor',
                    description='ACE inhibitor or ARB therapy',
                    risk_reduction=0.30,  # 30% slower progression
                    time_to_effect=180,
                    confidence=0.92,
                    source='Jafar et al, Ann Intern Med 2001;135:73-87'
                ),
                InterventionImpact(
                    intervention='protein_restriction',
                    description='Reduce protein to 0.8g/kg/day',
                    risk_reduction=0.25,  # 25% slower progression
                    time_to_effect=180,
                    confidence=0.75,
                    source='Kasiske et al, Am J Kidney Dis 1998;31:954-961'
                ),
            ],
        }
    
    def calculate_intervention_impact(
        self,
        disease: str,
        current_risk: float,
        interventions: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate combined impact of multiple interventions
        
        Args:
            disease: Disease name (diabetes, cardiovascular, etc.)
            current_risk: Current risk probability (0-1)
            interventions: List of intervention names to apply
        
        Returns:
            Dict with new risk, reduction, and details
        """
        if disease not in self.interventions:
            return {
                'error': f'Unknown disease: {disease}',
                'current_risk': current_risk,
                'new_risk': current_risk
            }
        
        available_interventions = {
            i.intervention: i for i in self.interventions[disease]
        }
        
        # Calculate combined effect (multiplicative model)
        combined_reduction = 1.0
        applied_interventions = []
        
        for intervention_name in interventions:
            if intervention_name in available_interventions:
                intervention = available_interventions[intervention_name]
                combined_reduction *= (1 - intervention.risk_reduction)
                applied_interventions.append(intervention)
        
        # New risk after interventions
        new_risk = current_risk * combined_reduction
        absolute_reduction = current_risk - new_risk
        relative_reduction = 1 - combined_reduction
        
        return {
            'disease': disease,
            'current_risk': current_risk,
            'new_risk': new_risk,
            'absolute_reduction': absolute_reduction,
            'relative_reduction': relative_reduction,
            'interventions_applied': applied_interventions,
            'time_to_full_effect': max([i.time_to_effect for i in applied_interventions]) if applied_interventions else 0
        }
    
    def recommend_interventions(
        self,
        disease: str,
        current_risk: float,
        target_risk: float = 0.1,
        max_interventions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recommend best interventions to reach target risk
        
        Returns list of intervention combinations ranked by effectiveness
        """
        if disease not in self.interventions:
            return []
        
        available = self.interventions[disease]
        
        # Sort by effectiveness (risk_reduction * confidence)
        sorted_interventions = sorted(
            available,
            key=lambda x: x.risk_reduction * x.confidence,
            reverse=True
        )
        
        recommendations = []
        
        # Try combinations of increasing size
        for n in range(1, min(max_interventions + 1, len(sorted_interventions) + 1)):
            from itertools import combinations
            for combo in combinations(sorted_interventions, n):
                intervention_names = [i.intervention for i in combo]
                result = self.calculate_intervention_impact(
                    disease, current_risk, intervention_names
                )
                
                if result['new_risk'] <= target_risk:
                    recommendations.append({
                        'interventions': combo,
                        'result': result,
                        'feasibility': self._assess_feasibility(combo)
                    })
        
        # Sort by feasibility and effectiveness
        recommendations.sort(
            key=lambda x: (x['feasibility'], -x['result']['new_risk']),
            reverse=True
        )
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _assess_feasibility(self, interventions: Tuple[InterventionImpact]) -> float:
        """Assess how feasible a combination of interventions is"""
        # Lifestyle changes are more feasible than medications
        feasibility_scores = {
            'exercise_increase': 0.8,
            'weight_loss': 0.7,
            'mediterranean_diet': 0.8,
            'dash_diet': 0.8,
            'sleep_improvement': 0.9,
            'stress_reduction': 0.7,
            'sodium_reduction': 0.8,
            'smoking_cessation': 0.6,  # Hard but important
            'protein_restriction': 0.7,
            'metformin': 0.9,  # Easy to take
            'statin_therapy': 0.9,
            'ace_inhibitor': 0.9,
            'bp_control': 0.8,
            'glucose_control': 0.7,
        }
        
        scores = [feasibility_scores.get(i.intervention, 0.5) for i in interventions]
        return np.mean(scores) if scores else 0.5
    
    def get_specific_recommendation(
        self,
        disease: str,
        current_lifestyle: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations based on current lifestyle
        """
        recommendations = []
        
        if disease not in self.interventions:
            return recommendations
        
        # Check what can be improved
        exercise_sessions = current_lifestyle.get('exercise_sessions_per_week', 1)
        sleep_hours = current_lifestyle.get('sleep_hours', 6.5)
        bmi = current_lifestyle.get('bmi', 25)
        diet_quality = current_lifestyle.get('diet_quality', 'moderate')
        
        for intervention in self.interventions[disease]:
            applicable = False
            specific_advice = ""
            
            if intervention.intervention == 'exercise_increase' and exercise_sessions < 5:
                applicable = True
                current_minutes = exercise_sessions * 30  # Assume 30 min per session
                target_minutes = 150
                increase_needed = max(0, target_minutes - current_minutes)
                specific_advice = f"Increase exercise from {current_minutes} to {target_minutes} min/week (+{increase_needed} min)"
            
            elif intervention.intervention == 'sleep_improvement' and sleep_hours < 7:
                applicable = True
                increase_needed = 7.5 - sleep_hours
                specific_advice = f"Increase sleep from {sleep_hours:.1f} to 7-8 hours/night (+{increase_needed:.1f}h)"
            
            elif intervention.intervention == 'weight_loss' and bmi > 25:
                applicable = True
                target_bmi = 24
                specific_advice = f"Reduce BMI from {bmi:.1f} to {target_bmi} (7% body weight loss)"
            
            elif intervention.intervention in ['mediterranean_diet', 'dash_diet'] and diet_quality == 'poor':
                applicable = True
                specific_advice = f"Improve diet quality from {diet_quality} to good"
            
            elif intervention.intervention == 'stress_reduction':
                applicable = True
                specific_advice = "Add stress management: 20 min meditation or yoga daily"
            
            if applicable:
                recommendations.append({
                    'intervention': intervention,
                    'specific_advice': specific_advice,
                    'impact': f"{intervention.risk_reduction:.0%} risk reduction",
                    'time_frame': f"{intervention.time_to_effect} days to full effect",
                    'confidence': f"{intervention.confidence:.0%} evidence confidence"
                })
        
        return recommendations


# Global instance
_calculator = None

def get_intervention_calculator() -> InterventionCalculator:
    """Get or create global intervention calculator"""
    global _calculator
    if _calculator is None:
        _calculator = InterventionCalculator()
    return _calculator
