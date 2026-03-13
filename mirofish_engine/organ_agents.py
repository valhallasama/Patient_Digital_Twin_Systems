"""
Specific Organ/System Agents - FIXED VERSION with realistic disease emergence
Each with unique physiological logic and interaction patterns
"""

from typing import Dict, List, Any
import numpy as np
from .body_system_agent import BodySystemAgent, AgentPersonality


class CardiovascularAgent(BodySystemAgent):
    """Heart and blood vessels - pumps blood, regulates pressure"""
    
    def __init__(self, initial_state: Dict[str, Any], personality: AgentPersonality = None):
        super().__init__("Cardiovascular", initial_state, personality)
        
        # Specific state variables
        self.state.setdefault('systolic_bp', 120)
        self.state.setdefault('diastolic_bp', 80)
        self.state.setdefault('heart_rate', 70)
        self.state.setdefault('cardiac_output', 5.0)  # L/min
        self.state.setdefault('vessel_elasticity', 1.0)
        self.state.setdefault('atherosclerosis_level', 0.0)
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        """Cardiovascular-specific decision logic"""
        decision = super().decide(perceptions)
        
        # Get signals from environment
        signals = perceptions.get('signals_for_me', {})
        
        # Respond to glucose levels (from Metabolic) - FIXED: lower threshold, faster damage
        glucose = signals.get('glucose', 5.0)
        if glucose > 6.5:  # Lower threshold
            # High glucose damages vessels
            self.state['vessel_elasticity'] *= 0.9995  # Faster decline
            self.state['atherosclerosis_level'] += 0.002  # Faster accumulation
            self.stress_level += 0.015
            decision['signals_to_send']['vessel_damage'] = True
        
        # Respond to stress hormones (from Endocrine) - FIXED: lower threshold
        cortisol = signals.get('cortisol', 1.0)
        if cortisol > 1.2:  # Lower threshold
            # Stress increases BP and HR
            bp_increase = 1.5 if cortisol > 1.5 else 0.5
            self.state['systolic_bp'] += bp_increase
            self.state['heart_rate'] += 3
            self.stress_level += 0.01
            decision['signals_to_send']['elevated_bp'] = self.state['systolic_bp']
        
        # Respond to inflammation (from Immune) - FIXED: lower threshold
        inflammation = signals.get('inflammation', 0.0)
        if inflammation > 0.3:  # Lower threshold
            self.state['atherosclerosis_level'] += 0.003
            self.stress_level += 0.01
        
        # LDL contributes to atherosclerosis - NEW
        ldl = signals.get('ldl', 2.5)
        if ldl > 3.5:
            self.state['atherosclerosis_level'] += 0.002
        
        # Bounds to prevent unrealistic values
        self.state['systolic_bp'] = max(80, min(200, self.state['systolic_bp']))
        self.state['diastolic_bp'] = max(50, min(120, self.state['diastolic_bp']))
        self.state['heart_rate'] = max(40, min(150, self.state['heart_rate']))
        self.state['vessel_elasticity'] = max(0.1, min(1.0, self.state['vessel_elasticity']))
        self.state['atherosclerosis_level'] = max(0, min(10.0, self.state['atherosclerosis_level']))
        
        # Send oxygen and nutrients to all systems
        decision['signals_to_send']['oxygen_delivery'] = self.state['cardiac_output']
        decision['signals_to_send']['blood_pressure'] = self.state['systolic_bp']
        decision['signals_to_send']['bp'] = self.state['systolic_bp']
        decision['signals_to_send']['vessel_health'] = self.state['vessel_elasticity']
        
        return decision
    
    def _summarize_state(self) -> str:
        return f"BP: {self.state['systolic_bp']:.0f}/{self.state['diastolic_bp']:.0f}, HR: {self.state['heart_rate']:.0f}, Vessel health: {self.state['vessel_elasticity']:.2f}"


class MetabolicAgent(BodySystemAgent):
    """Pancreas and metabolic regulation - manages glucose and insulin"""
    
    def __init__(self, initial_state: Dict[str, Any], personality: AgentPersonality = None):
        super().__init__("Metabolic", initial_state, personality)
        
        self.state.setdefault('glucose', 5.0)  # mmol/L
        self.state.setdefault('insulin', 10.0)  # μU/mL
        self.state.setdefault('hba1c', 5.0)  # %
        self.state.setdefault('insulin_sensitivity', 1.0)
        self.state.setdefault('beta_cell_function', 1.0)
        self.state.setdefault('insulin_resistance', 0.0)
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        decision = super().decide(perceptions)
        signals = perceptions.get('signals_for_me', {})
        
        # Respond to food intake
        food_glucose = signals.get('food_glucose', 0.0)
        if food_glucose > 0:
            self.state['glucose'] += food_glucose
            # Produce insulin to handle glucose
            insulin_needed = food_glucose * 2.0 / self.state['insulin_sensitivity']
            self.state['insulin'] += insulin_needed
            
            # Beta cells get stressed if overworked - FIXED: faster decline
            if insulin_needed > 50:
                self.state['beta_cell_function'] *= 0.9995  # Faster decline
                self.stress_level += 0.02
        
        # Respond to stress hormones - FIXED: lower threshold, faster damage
        cortisol = signals.get('cortisol', 1.0)
        if cortisol > 1.2:  # Lower threshold
            # Chronic cortisol exposure damages insulin sensitivity
            decline_rate = 0.9995 if cortisol > 1.5 else 0.9998
            self.state['insulin_sensitivity'] *= decline_rate
            self.state['insulin_resistance'] = 1.0 - self.state['insulin_sensitivity']
            self.stress_level += 0.015
            decision['signals_to_send']['insulin_resistance_rising'] = True
        
        # Chronic high glucose damages beta cells - NEW
        if self.state['glucose'] > 6.0:
            self.state['beta_cell_function'] *= 0.9998
            self.stress_level += 0.01
        
        # Glucose regulation
        if self.state['glucose'] > 5.5:
            # Try to lower glucose
            glucose_lowered = min(
                self.state['glucose'] - 5.0,
                self.state['insulin'] * 0.1 * self.state['insulin_sensitivity']
            )
            self.state['glucose'] -= glucose_lowered
            self.state['insulin'] -= glucose_lowered * 10
            
            # If can't lower glucose effectively, stress increases - NEW
            if glucose_lowered < 0.1:
                self.stress_level += 0.02
        
        # Update HbA1c (slow-moving average, faster accumulation) - FIXED
        self.state['hba1c'] = self.state['hba1c'] * 0.997 + self.state['glucose'] * 0.003
        
        # Bounds to prevent unrealistic values
        self.state['glucose'] = max(3.0, min(25.0, self.state['glucose']))
        self.state['hba1c'] = max(4.0, min(15.0, self.state['hba1c']))
        self.state['insulin'] = max(0, min(200, self.state['insulin']))
        self.state['insulin_sensitivity'] = max(0.1, min(1.0, self.state['insulin_sensitivity']))
        self.state['beta_cell_function'] = max(0.1, min(1.0, self.state['beta_cell_function']))
        
        # Send glucose to all systems
        decision['signals_to_send']['glucose'] = self.state['glucose']
        decision['signals_to_send']['insulin_resistance'] = self.state['insulin_resistance']
        
        # Alert if diabetes emerging - FIXED: earlier warning
        if self.state['hba1c'] > 6.5:
            decision['signals_to_send']['diabetes_alert'] = True
            self.stress_level = 0.8
        elif self.state['hba1c'] > 6.0:
            self.stress_level = max(self.stress_level, 0.5)
        
        return decision
    
    def _summarize_state(self) -> str:
        return f"Glucose: {self.state['glucose']:.1f}, HbA1c: {self.state['hba1c']:.1f}%, Insulin sensitivity: {self.state['insulin_sensitivity']:.2f}"


class RenalAgent(BodySystemAgent):
    """Kidneys - filter blood, regulate electrolytes"""
    
    def __init__(self, initial_state: Dict[str, Any], personality: AgentPersonality = None):
        super().__init__("Renal", initial_state, personality)
        
        self.state.setdefault('egfr', 100)  # mL/min
        self.state.setdefault('creatinine', 80)  # μmol/L
        self.state.setdefault('filtration_capacity', 1.0)
        self.state.setdefault('damage_level', 0.0)
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        decision = super().decide(perceptions)
        signals = perceptions.get('signals_for_me', {})
        
        # Respond to blood pressure
        bp = signals.get('blood_pressure', 120)
        if bp > 140:
            # High BP damages kidneys
            self.state['damage_level'] += 0.001
            self.state['filtration_capacity'] *= 0.999
            self.stress_level += 0.01
        
        # Respond to high glucose (diabetic nephropathy)
        glucose = signals.get('glucose', 5.0)
        if glucose > 7.0:
            self.state['damage_level'] += 0.002
            self.state['filtration_capacity'] *= 0.998
        
        # Progressive damage if under stress
        if self.stress_level > 0.5:
            self.state['damage_level'] += 0.001
            self.state['filtration_capacity'] *= 0.9999
        
        # Bounds to prevent unrealistic values
        self.state['egfr'] = max(5, min(120, self.state['egfr']))
        self.state['creatinine'] = max(50, min(1000, self.state['creatinine']))
        self.state['filtration_capacity'] = max(0.05, min(1.0, self.state['filtration_capacity']))
        self.state['damage_level'] = max(0, min(10.0, self.state['damage_level']))
        
        # Send renal status
        decision['signals_to_send']['kidney_function'] = self.state['egfr'] = 100 * self.state['filtration_capacity']
        self.state['creatinine'] = 80 / self.state['filtration_capacity']
        
        # Send signals
        decision['signals_to_send']['egfr'] = self.state['egfr']
        decision['signals_to_send']['kidney_function'] = self.state['filtration_capacity']
        
        if self.state['egfr'] < 60:
            decision['signals_to_send']['ckd_alert'] = True
            self.stress_level = 0.7
        
        return decision
    
    def _summarize_state(self) -> str:
        return f"eGFR: {self.state['egfr']:.0f}, Creatinine: {self.state['creatinine']:.0f}, Function: {self.state['filtration_capacity']:.2f}"


class HepaticAgent(BodySystemAgent):
    """Liver - metabolizes nutrients, produces proteins"""
    
    def __init__(self, initial_state: Dict[str, Any], personality: AgentPersonality = None):
        super().__init__("Hepatic", initial_state, personality)
        
        self.state.setdefault('alt', 25)  # U/L
        self.state.setdefault('ast', 25)  # U/L
        self.state.setdefault('ldl', 2.5)  # mmol/L
        self.state.setdefault('hdl', 1.5)  # mmol/L
        self.state.setdefault('fat_content', 0.0)  # 0-1
        self.state.setdefault('detox_capacity', 1.0)
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        decision = super().decide(perceptions)
        signals = perceptions.get('signals_for_me', {})
        
        # Process dietary fats
        dietary_fat = signals.get('dietary_fat', 0.0)
        if dietary_fat > 50:  # High fat intake
            self.state['fat_content'] += 0.001
            self.state['ldl'] += 0.01
            if self.state['fat_content'] > 0.05:
                # Fatty liver
                self.state['alt'] += 1
                self.state['ast'] += 0.5
                self.stress_level += 0.01
        
        # Respond to inflammation
        inflammation = signals.get('inflammation', 0.0)
        # Progressive fat accumulation if poor diet
        if self.stress_level > 0.4:
            self.state['fat_content'] += 0.001
        
        # Bounds to prevent unrealistic values
        self.state['alt'] = max(10, min(500, self.state['alt']))
        self.state['ast'] = max(10, min(400, self.state['ast']))
        self.state['ldl'] = max(1.0, min(10.0, self.state['ldl']))
        self.state['hdl'] = max(0.5, min(3.0, self.state['hdl']))
        self.state['fat_content'] = max(0, min(1.0, self.state['fat_content']))
        self.state['detox_capacity'] = max(0.1, min(1.0, self.state['detox_capacity']))
        
        # Send hepatic status
        decision['signals_to_send']['ldl'] = self.state['ldl']
        decision['signals_to_send']['liver_health'] = self.state['detox_capacity']
        
        if self.state['ldl'] > 4.0:
            decision['signals_to_send']['high_ldl_alert'] = True
        
        return decision
    
    def _summarize_state(self) -> str:
        return f"ALT: {self.state['alt']:.0f}, LDL: {self.state['ldl']:.1f}, Fat: {self.state['fat_content']:.1%}"


class ImmuneAgent(BodySystemAgent):
    """Immune system - fights infections, manages inflammation"""
    
    def __init__(self, initial_state: Dict[str, Any], personality: AgentPersonality = None):
        super().__init__("Immune", initial_state, personality)
        
        self.state.setdefault('wbc', 7.0)  # ×10⁹/L
        self.state.setdefault('crp', 1.0)  # mg/L
        self.state.setdefault('inflammation', 0.0)  # 0-1
        self.state.setdefault('immune_activation', 0.0)
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        decision = super().decide(perceptions)
        signals = perceptions.get('signals_for_me', {})
        
        # Detect damage signals from other agents
        vessel_damage = signals.get('vessel_damage', False)
        insulin_resistance = signals.get('insulin_resistance_rising', False)
        
        if vessel_damage or insulin_resistance:
            # Activate immune response
            self.state['inflammation'] += 0.01
            self.state['crp'] += 0.1
            self.state['immune_activation'] += 0.05
        
        # Chronic inflammation
        if self.state['inflammation'] > 0.5:
            self.stress_level += 0.02
            # Inflammation damages beta cells
            decision['signals_to_send']['chronic_inflammation'] = True
        
        # Natural resolution of inflammation
        self.state['inflammation'] *= 0.98
        self.state['crp'] = max(0.5, self.state['crp'] * 0.99)
        
        decision['signals_to_send']['inflammation'] = self.state['inflammation']
        
        return decision
    
    def _summarize_state(self) -> str:
        return f"WBC: {self.state['wbc']:.1f}, CRP: {self.state['crp']:.1f}, Inflammation: {self.state['inflammation']:.2f}"


class EndocrineAgent(BodySystemAgent):
    """Endocrine system - hormonal regulation"""
    
    def __init__(self, initial_state: Dict[str, Any], personality: AgentPersonality = None):
        super().__init__("Endocrine", initial_state, personality)
        
        self.state.setdefault('cortisol', 1.0)  # Relative units
        self.state.setdefault('thyroid', 1.0)
        self.state.setdefault('stress_response', 0.0)
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        decision = super().decide(perceptions)
        signals = perceptions.get('signals_for_me', {})
        
        # Respond to stress from Neural agent - FIXED: lower threshold, slower recovery
        stress_signal = signals.get('stress_level', 0.0)
        if stress_signal > 0.4:  # Lower threshold
            self.state['cortisol'] += 0.08
            self.state['stress_response'] = stress_signal
        else:
            # Cortisol returns to baseline slowly
            self.state['cortisol'] = max(1.0, self.state['cortisol'] * 0.98)
        
        # Send hormones to all systems
        decision['signals_to_send']['cortisol'] = self.state['cortisol']
        decision['signals_to_send']['thyroid'] = self.state['thyroid']
        
        return decision
    
    def _summarize_state(self) -> str:
        return f"Cortisol: {self.state['cortisol']:.2f}, Stress response: {self.state['stress_response']:.2f}"


class NeuralAgent(BodySystemAgent):
    """Brain and nervous system - controls and coordinates"""
    
    def __init__(self, initial_state: Dict[str, Any], personality: AgentPersonality = None):
        super().__init__("Neural", initial_state, personality)
        
        self.state.setdefault('stress_level', 0.0)
        self.state.setdefault('sleep_quality', 0.8)
        self.state.setdefault('cognitive_function', 1.0)
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        decision = super().decide(perceptions)
        signals = perceptions.get('signals_for_me', {})
        
        # Detect stress from lifestyle events
        lifestyle_stress = signals.get('lifestyle_stress', 0.0)
        self.state['stress_level'] = lifestyle_stress
        
        # Poor sleep increases stress - FIXED: stronger effect
        if self.state['sleep_quality'] < 0.7:
            self.state['stress_level'] += 0.15
        
        # Maintain baseline stress if sedentary/poor lifestyle - NEW
        if self.state['stress_level'] < 0.4:
            self.state['stress_level'] = 0.4  # Chronic baseline stress
        
        # Send stress signals to endocrine
        decision['signals_to_send']['stress_level'] = self.state['stress_level']
        
        return decision
    
    def _summarize_state(self) -> str:
        return f"Stress: {self.state['stress_level']:.2f}, Sleep: {self.state['sleep_quality']:.2f}"


def create_agent_from_seed(agent_type: str, seed_data: Dict[str, Any]) -> BodySystemAgent:
    """Factory function to create agents from seed information"""
    
    personality = AgentPersonality(
        resilience=seed_data.get('resilience', 0.5),
        reactivity=seed_data.get('reactivity', 0.5),
        adaptability=seed_data.get('adaptability', 0.5),
        cooperation=seed_data.get('cooperation', 0.5)
    )
    
    agent_classes = {
        'cardiovascular': CardiovascularAgent,
        'metabolic': MetabolicAgent,
        'renal': RenalAgent,
        'hepatic': HepaticAgent,
        'immune': ImmuneAgent,
        'endocrine': EndocrineAgent,
        'neural': NeuralAgent
    }
    
    agent_class = agent_classes.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_class(seed_data.get('initial_state', {}), personality)
