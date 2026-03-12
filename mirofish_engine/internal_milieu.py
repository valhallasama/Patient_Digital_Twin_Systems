"""
Internal Milieu - The shared environment for all body system agents
Analogous to MiroFish's digital world environment
"""

from typing import Dict, List, Any
from datetime import datetime
import numpy as np


class InternalMilieu:
    """
    Shared internal environment where all agents interact
    Blood composition, hormones, nutrients, signals
    """
    
    def __init__(self, initial_composition: Dict[str, Any] = None):
        self.composition = initial_composition or {}
        self.signals = {}  # Agent-to-agent signals
        self.timestamp = datetime.now()
        self.day = 0
        
        # Default composition
        self.composition.setdefault('glucose', 5.0)
        self.composition.setdefault('oxygen', 95.0)  # %
        self.composition.setdefault('cortisol', 1.0)
        self.composition.setdefault('insulin', 10.0)
        self.composition.setdefault('inflammation_markers', 0.0)
        self.composition.setdefault('ldl', 2.5)
        self.composition.setdefault('hdl', 1.5)
        
        # External inputs (lifestyle)
        self.external_inputs = {
            'food_intake': 0.0,
            'exercise': 0.0,
            'stress': 0.0,
            'sleep_quality': 0.8,
            'medications': []
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return {
            'composition': self.composition.copy(),
            'external_inputs': self.external_inputs.copy(),
            'day': self.day,
            'timestamp': self.timestamp
        }
    
    def get_signals_for(self, agent_name: str) -> Dict[str, Any]:
        """Get signals relevant to specific agent"""
        # Return composition + signals directed to this agent
        relevant_signals = self.composition.copy()
        
        if agent_name in self.signals:
            relevant_signals.update(self.signals[agent_name])
        
        # Add external inputs
        relevant_signals.update(self.external_inputs)
        
        return relevant_signals
    
    def get_other_agents_state(self, requesting_agent: str) -> Dict[str, Any]:
        """Get state of other agents (for awareness)"""
        # Simplified - in full implementation, would query actual agents
        return {
            'other_agents_stress': self.composition.get('system_stress', 0.0)
        }
    
    def update(self, agent_decisions: Dict[str, Dict[str, Any]], 
               agent_interactions: List[Dict[str, Any]]) -> None:
        """
        Update environment based on all agent decisions and interactions
        This is where swarm intelligence emerges
        """
        # Clear old signals
        self.signals = {}
        
        # Process each agent's decision
        for agent_name, decision in agent_decisions.items():
            signals_to_send = decision.get('signals_to_send', {})
            
            # Update composition with agent outputs
            for signal_name, signal_value in signals_to_send.items():
                if signal_name in ['glucose', 'insulin', 'cortisol', 'ldl', 'hdl', 'inflammation']:
                    # These affect global composition
                    self.composition[signal_name] = signal_value
                else:
                    # These are targeted signals
                    # Store for specific agents to receive
                    self._route_signal(signal_name, signal_value)
        
        # Process agent-to-agent interactions
        for interaction in agent_interactions:
            from_agent = interaction['from']
            to_agent = interaction['to']
            content = interaction['content']
            
            if to_agent not in self.signals:
                self.signals[to_agent] = {}
            
            self.signals[to_agent][f'{from_agent}_message'] = content
        
        # Natural decay/regulation
        self._apply_homeostasis()
        
        # Advance time
        self.day += 1
        self.timestamp = datetime.now()
    
    def apply_intervention(self, intervention: Dict[str, Any]) -> None:
        """Apply external intervention (medication, lifestyle change)"""
        intervention_type = intervention.get('type')
        
        if intervention_type == 'medication':
            drug = intervention.get('drug')
            dose = intervention.get('dose', 1.0)
            
            if drug == 'metformin':
                # Improves insulin sensitivity
                self.composition['insulin_sensitivity_modifier'] = 1.2 * dose
            elif drug == 'statin':
                # Lowers LDL
                self.composition['ldl'] *= (1.0 - 0.3 * dose)
            elif drug == 'antihypertensive':
                # Lowers BP
                self.composition['bp_modifier'] = 0.9 * dose
            
            self.external_inputs['medications'].append(drug)
        
        elif intervention_type == 'lifestyle':
            change = intervention.get('change')
            intensity = intervention.get('intensity', 1.0)
            
            if change == 'exercise':
                self.external_inputs['exercise'] = intensity
                # Exercise improves insulin sensitivity, reduces stress
                self.composition['insulin_sensitivity_modifier'] = 1.1 * intensity
                self.external_inputs['stress'] *= (1.0 - 0.2 * intensity)
            
            elif change == 'diet':
                diet_type = intervention.get('diet_type', 'balanced')
                if diet_type == 'low_carb':
                    self.external_inputs['food_intake'] *= 0.7
                elif diet_type == 'mediterranean':
                    self.composition['ldl'] *= 0.9
                    self.composition['inflammation_markers'] *= 0.8
            
            elif change == 'stress_reduction':
                self.external_inputs['stress'] *= (1.0 - intensity)
            
            elif change == 'sleep_improvement':
                self.external_inputs['sleep_quality'] = min(1.0, 
                    self.external_inputs['sleep_quality'] + 0.1 * intensity)
    
    def _route_signal(self, signal_name: str, signal_value: Any) -> None:
        """Route signal to appropriate agents"""
        # Determine which agents should receive this signal
        signal_routing = {
            'blood_pressure': ['renal', 'cardiovascular'],
            'insulin_resistance': ['metabolic', 'hepatic'],
            'vessel_damage': ['immune', 'cardiovascular'],
            'kidney_function': ['cardiovascular', 'endocrine'],
            'diabetes_alert': ['cardiovascular', 'renal', 'neural'],
            'high_ldl_alert': ['cardiovascular', 'immune']
        }
        
        target_agents = signal_routing.get(signal_name, [])
        
        for agent in target_agents:
            if agent not in self.signals:
                self.signals[agent] = {}
            self.signals[agent][signal_name] = signal_value
    
    def _apply_homeostasis(self) -> None:
        """Natural regulatory mechanisms"""
        # Glucose regulation
        if self.composition['glucose'] > 5.5:
            self.composition['glucose'] *= 0.99
        
        # Cortisol circadian rhythm
        self.composition['cortisol'] = max(1.0, self.composition['cortisol'] * 0.98)
        
        # Inflammation resolution
        if 'inflammation_markers' in self.composition:
            self.composition['inflammation_markers'] *= 0.95
    
    def get_summary(self) -> str:
        """Get human-readable summary of environment"""
        return f"""
Internal Milieu (Day {self.day}):
  Glucose: {self.composition.get('glucose', 0):.1f} mmol/L
  Cortisol: {self.composition.get('cortisol', 0):.2f}
  LDL: {self.composition.get('ldl', 0):.1f} mmol/L
  Inflammation: {self.composition.get('inflammation_markers', 0):.2f}
  
External Inputs:
  Exercise: {self.external_inputs.get('exercise', 0):.1f}
  Stress: {self.external_inputs.get('stress', 0):.1f}
  Sleep Quality: {self.external_inputs.get('sleep_quality', 0):.1f}
  Medications: {', '.join(self.external_inputs.get('medications', [])) or 'None'}
"""
