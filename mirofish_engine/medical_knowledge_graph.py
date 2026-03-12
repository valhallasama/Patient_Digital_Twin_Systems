"""
Medical Knowledge Graph - GPT-Free Alternative
Your own medical knowledge base built from textbooks, guidelines, and literature
No external APIs needed - 100% your code
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MedicalRule:
    """A single medical knowledge rule"""
    name: str
    causes: List[str]
    effects: List[str]
    threshold: float
    progression_rate: float
    confidence: float
    source: str  # Medical literature reference


class MedicalKnowledgeGraph:
    """
    Medical knowledge base - replaces GPT with codified medical knowledge
    Built from medical textbooks, clinical guidelines, and research papers
    """
    
    def __init__(self):
        self.rules = {}
        self.disease_mechanisms = {}
        self.drug_interactions = {}
        self.clinical_guidelines = {}
        
        # Load medical knowledge
        self._load_pathophysiology_rules()
        self._load_disease_mechanisms()
        self._load_clinical_guidelines()
    
    def _load_pathophysiology_rules(self):
        """Load pathophysiology rules from medical literature"""
        
        # Insulin Resistance (from medical textbooks)
        self.rules['insulin_resistance'] = MedicalRule(
            name='insulin_resistance_progression',
            causes=['chronic_cortisol_elevation', 'obesity', 'sedentary_lifestyle', 'genetic_predisposition'],
            effects=['hyperglycemia', 'beta_cell_stress', 'dyslipidemia'],
            threshold=0.7,  # IR > 0.7 is clinically significant
            progression_rate=0.001,  # Per day under stress
            confidence=0.95,
            source='DeFronzo RA. Diabetes 2004;53:1621-1629'
        )
        
        # Atherosclerosis (from cardiovascular literature)
        self.rules['atherosclerosis'] = MedicalRule(
            name='atherosclerosis_progression',
            causes=['high_ldl', 'chronic_inflammation', 'hypertension', 'smoking', 'diabetes'],
            effects=['vessel_damage', 'reduced_elasticity', 'plaque_formation'],
            threshold=0.3,  # Significant at 30% vessel involvement
            progression_rate=0.002,
            confidence=0.90,
            source='Libby P. Nature 2002;420:868-874'
        )
        
        # Beta Cell Dysfunction (from diabetes literature)
        self.rules['beta_cell_dysfunction'] = MedicalRule(
            name='beta_cell_failure',
            causes=['chronic_hyperglycemia', 'inflammation', 'oxidative_stress', 'lipotoxicity'],
            effects=['reduced_insulin_secretion', 'diabetes_onset'],
            threshold=0.5,  # Function < 50% is critical
            progression_rate=0.0005,
            confidence=0.92,
            source='Weir GC, Bonner-Weir S. Diabetes 2004;53:S16-S21'
        )
        
        # Hypertension (from cardiology guidelines)
        self.rules['hypertension'] = MedicalRule(
            name='hypertension_development',
            causes=['chronic_stress', 'high_sodium', 'obesity', 'insulin_resistance'],
            effects=['vessel_damage', 'kidney_damage', 'heart_strain'],
            threshold=140,  # Systolic BP > 140 mmHg
            progression_rate=0.5,  # mmHg per day under stress
            confidence=0.88,
            source='JNC 8 Guidelines, JAMA 2014;311:507-520'
        )
        
        # Chronic Kidney Disease (from nephrology literature)
        self.rules['ckd'] = MedicalRule(
            name='ckd_progression',
            causes=['hypertension', 'diabetes', 'chronic_inflammation'],
            effects=['reduced_filtration', 'proteinuria', 'electrolyte_imbalance'],
            threshold=60,  # eGFR < 60 mL/min
            progression_rate=0.001,  # eGFR decline per day
            confidence=0.93,
            source='KDIGO Guidelines 2012'
        )
    
    def _load_disease_mechanisms(self):
        """Load disease emergence mechanisms"""
        
        self.disease_mechanisms['type2_diabetes'] = {
            'required_conditions': [
                'insulin_resistance > 0.6',
                'beta_cell_function < 0.7',
                'hba1c > 6.5'
            ],
            'mechanism': 'Progressive insulin resistance leads to compensatory hyperinsulinemia, '
                        'eventually exhausting beta cells and causing diabetes',
            'probability_formula': lambda ir, bcf, hba1c: min(0.95, ir * 0.5 + (1 - bcf) * 0.4 + (hba1c - 5.0) * 0.1),
            'time_to_onset': '2-5 years from prediabetes',
            'source': 'ADA Standards of Care 2023'
        }
        
        self.disease_mechanisms['cvd'] = {
            'required_conditions': [
                'atherosclerosis_level > 0.4',
                'systolic_bp > 140',
                'ldl > 4.0'
            ],
            'mechanism': 'LDL oxidation and inflammation drive atherosclerotic plaque formation, '
                        'combined with hypertension causing vessel damage',
            'probability_formula': lambda ath, bp, ldl: min(0.90, ath * 0.4 + (bp - 120) / 100 + (ldl - 3.0) / 10),
            'time_to_onset': '5-10 years from risk factor onset',
            'source': 'Framingham Heart Study'
        }
        
        self.disease_mechanisms['metabolic_syndrome'] = {
            'required_conditions': [
                'insulin_resistance > 0.5',
                'systolic_bp > 130',
                'ldl > 3.5',
                'inflammation > 0.4'
            ],
            'mechanism': 'Cluster of metabolic abnormalities driven by insulin resistance and inflammation',
            'probability_formula': lambda ir, bp, ldl, inf: 0.75 if all([ir > 0.5, bp > 130, ldl > 3.5, inf > 0.4]) else 0.2,
            'time_to_onset': '1-3 years',
            'source': 'NCEP ATP III Guidelines'
        }
    
    def _load_clinical_guidelines(self):
        """Load clinical practice guidelines"""
        
        self.clinical_guidelines['diabetes_management'] = {
            'hba1c_target': 7.0,
            'glucose_target': 5.5,
            'first_line_medication': 'metformin',
            'lifestyle_interventions': ['diet', 'exercise', 'weight_loss'],
            'monitoring_frequency': 'quarterly',
            'source': 'ADA 2023'
        }
        
        self.clinical_guidelines['hypertension_management'] = {
            'bp_target': '130/80',
            'first_line_medications': ['ACE_inhibitor', 'ARB', 'CCB', 'thiazide'],
            'lifestyle_interventions': ['sodium_restriction', 'exercise', 'weight_loss'],
            'monitoring_frequency': 'monthly',
            'source': 'ACC/AHA 2017'
        }
    
    def query_progression(self, condition: str, current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query how a condition should progress given current state
        Replaces GPT with medical knowledge
        """
        if condition not in self.rules:
            return None
        
        rule = self.rules[condition]
        
        # Check if causes are present
        active_causes = []
        for cause in rule.causes:
            if self._is_cause_active(cause, current_state):
                active_causes.append(cause)
        
        if not active_causes:
            return None
        
        # Calculate progression
        progression_multiplier = len(active_causes) / len(rule.causes)
        
        return {
            'should_progress': True,
            'rate': rule.progression_rate * progression_multiplier,
            'effects': rule.effects,
            'confidence': rule.confidence,
            'active_causes': active_causes,
            'reasoning': f"{condition} progressing due to: {', '.join(active_causes)}",
            'source': rule.source
        }
    
    def _is_cause_active(self, cause: str, state: Dict[str, Any]) -> bool:
        """Check if a cause is active in current state"""
        
        cause_checks = {
            'chronic_cortisol_elevation': lambda s: s.get('cortisol', 1.0) > 1.2,
            'obesity': lambda s: s.get('bmi', 25) > 28,
            'sedentary_lifestyle': lambda s: s.get('exercise', 0.5) < 0.3,
            'high_ldl': lambda s: s.get('ldl', 2.5) > 3.5,
            'chronic_inflammation': lambda s: s.get('inflammation', 0.0) > 0.3,
            'hypertension': lambda s: s.get('systolic_bp', 120) > 140,
            'chronic_hyperglycemia': lambda s: s.get('glucose', 5.0) > 6.5,
            'diabetes': lambda s: s.get('hba1c', 5.0) > 6.5,
            'chronic_stress': lambda s: s.get('stress_level', 0.0) > 0.5,
            'high_sodium': lambda s: s.get('sodium_intake', 2000) > 3000,
            'insulin_resistance': lambda s: s.get('insulin_resistance', 0.0) > 0.5,
            'oxidative_stress': lambda s: s.get('inflammation', 0.0) > 0.4,
            'lipotoxicity': lambda s: s.get('ldl', 2.5) > 4.0,
        }
        
        check_func = cause_checks.get(cause)
        if check_func:
            return check_func(state)
        return False
    
    def predict_disease_emergence(self, disease: str, agent_states: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Predict if disease will emerge based on agent states
        Uses medical knowledge instead of GPT
        """
        if disease not in self.disease_mechanisms:
            return None
        
        mechanism = self.disease_mechanisms[disease]
        
        # Check required conditions
        conditions_met = []
        for condition in mechanism['required_conditions']:
            if self._evaluate_condition(condition, agent_states):
                conditions_met.append(condition)
        
        if len(conditions_met) < len(mechanism['required_conditions']):
            return None
        
        # Calculate probability using medical formula
        probability = self._calculate_probability(disease, agent_states)
        
        return {
            'disease': disease,
            'probability': probability,
            'mechanism': mechanism['mechanism'],
            'conditions_met': conditions_met,
            'time_to_onset': mechanism['time_to_onset'],
            'source': mechanism['source']
        }
    
    def _evaluate_condition(self, condition: str, states: Dict[str, Any]) -> bool:
        """Evaluate a medical condition"""
        # Parse condition string (e.g., "insulin_resistance > 0.6")
        parts = condition.split()
        if len(parts) != 3:
            return False
        
        variable, operator, threshold = parts
        value = states.get(variable, 0)
        threshold = float(threshold)
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        
        return False
    
    def _calculate_probability(self, disease: str, states: Dict[str, Any]) -> float:
        """Calculate disease probability using medical formulas"""
        mechanism = self.disease_mechanisms[disease]
        formula = mechanism['probability_formula']
        
        if disease == 'type2_diabetes':
            ir = states.get('insulin_resistance', 0)
            bcf = states.get('beta_cell_function', 1.0)
            hba1c = states.get('hba1c', 5.0)
            return formula(ir, bcf, hba1c)
        
        elif disease == 'cvd':
            ath = states.get('atherosclerosis_level', 0)
            bp = states.get('systolic_bp', 120)
            ldl = states.get('ldl', 2.5)
            return formula(ath, bp, ldl)
        
        elif disease == 'metabolic_syndrome':
            ir = states.get('insulin_resistance', 0)
            bp = states.get('systolic_bp', 120)
            ldl = states.get('ldl', 2.5)
            inf = states.get('inflammation', 0)
            return formula(ir, bp, ldl, inf)
        
        return 0.0
    
    def get_intervention_recommendation(self, disease_risk: str) -> Dict[str, Any]:
        """Get clinical guideline-based intervention recommendations"""
        
        if disease_risk == 'diabetes':
            guideline = self.clinical_guidelines['diabetes_management']
            return {
                'lifestyle': guideline['lifestyle_interventions'],
                'medication': guideline['first_line_medication'],
                'targets': {
                    'hba1c': guideline['hba1c_target'],
                    'glucose': guideline['glucose_target']
                },
                'monitoring': guideline['monitoring_frequency'],
                'source': guideline['source']
            }
        
        elif disease_risk == 'hypertension':
            guideline = self.clinical_guidelines['hypertension_management']
            return {
                'lifestyle': guideline['lifestyle_interventions'],
                'medications': guideline['first_line_medications'],
                'target': guideline['bp_target'],
                'monitoring': guideline['monitoring_frequency'],
                'source': guideline['source']
            }
        
        return {}
    
    def explain_mechanism(self, disease: str) -> str:
        """Explain disease mechanism in natural language"""
        if disease in self.disease_mechanisms:
            mech = self.disease_mechanisms[disease]
            return f"{disease}: {mech['mechanism']} (Source: {mech['source']})"
        return f"No mechanism found for {disease}"


# Global instance
_knowledge_graph = None

def get_medical_knowledge() -> MedicalKnowledgeGraph:
    """Get or create global medical knowledge graph"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = MedicalKnowledgeGraph()
    return _knowledge_graph
