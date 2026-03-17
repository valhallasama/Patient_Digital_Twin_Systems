#!/usr/bin/env python3
"""
Scenario and Intervention Simulation Engine
Enables "what-if" analysis for treatment planning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import copy
from mirofish_engine.digital_twin_simulator import DigitalTwinSimulator


class InterventionScenario:
    """Represents a specific intervention scenario"""
    
    def __init__(self, name: str, description: str):
        """
        Initialize intervention scenario
        
        Args:
            name: Scenario name
            description: Scenario description
        """
        self.name = name
        self.description = description
        self.lifestyle_changes = {}
        self.medication_effects = {}
        self.weight_change = 0.0
        self.duration_years = 5
    
    def set_lifestyle_change(self, factor: str, new_value: str):
        """Set lifestyle factor change"""
        self.lifestyle_changes[factor] = new_value
        return self
    
    def set_weight_change(self, kg_change: float):
        """Set target weight change in kg"""
        self.weight_change = kg_change
        return self
    
    def set_medication(self, med_type: str, effect: Dict):
        """Set medication effect"""
        self.medication_effects[med_type] = effect
        return self
    
    def set_duration(self, years: int):
        """Set simulation duration"""
        self.duration_years = years
        return self


class ScenarioSimulator:
    """
    Scenario simulation engine for intervention testing
    Enables "what-if" analysis for treatment planning
    """
    
    def __init__(self, baseline_patient_data: Dict):
        """
        Initialize scenario simulator
        
        Args:
            baseline_patient_data: Baseline patient data
        """
        self.baseline_data = baseline_patient_data
        self.scenarios: List[InterventionScenario] = []
        self.results = {}
    
    def add_scenario(self, scenario: InterventionScenario):
        """Add scenario to simulation"""
        self.scenarios.append(scenario)
        return self
    
    def create_baseline_scenario(self) -> InterventionScenario:
        """Create baseline (no intervention) scenario"""
        return InterventionScenario(
            name="Baseline",
            description="No lifestyle changes or interventions"
        )
    
    def create_lifestyle_intervention(self) -> InterventionScenario:
        """Create lifestyle intervention scenario"""
        scenario = InterventionScenario(
            name="Lifestyle Intervention",
            description="Improve diet, increase exercise, reduce stress"
        )
        scenario.set_lifestyle_change('physical_activity', 'moderate')
        scenario.set_lifestyle_change('diet_quality', 'good')
        scenario.set_lifestyle_change('stress_level', 'low')
        return scenario
    
    def create_weight_loss_scenario(self, kg_loss: float) -> InterventionScenario:
        """Create weight loss scenario"""
        scenario = InterventionScenario(
            name=f"Weight Loss ({kg_loss}kg)",
            description=f"Lose {kg_loss}kg through diet and exercise"
        )
        scenario.set_weight_change(-kg_loss)
        scenario.set_lifestyle_change('physical_activity', 'moderate')
        scenario.set_lifestyle_change('diet_quality', 'good')
        return scenario
    
    def create_medication_scenario(self, med_type: str) -> InterventionScenario:
        """Create medication intervention scenario"""
        scenarios_map = {
            'metformin': {
                'name': 'Metformin',
                'description': 'Metformin for glucose control',
                'effects': {'glucose_reduction': 20, 'hba1c_reduction': 0.5}
            },
            'statin': {
                'name': 'Statin Therapy',
                'description': 'Statin for cholesterol management',
                'effects': {'ldl_reduction': 30}
            },
            'ace_inhibitor': {
                'name': 'ACE Inhibitor',
                'description': 'ACE inhibitor for blood pressure',
                'effects': {'bp_reduction': 10}
            }
        }
        
        med_info = scenarios_map.get(med_type, {})
        scenario = InterventionScenario(
            name=med_info.get('name', med_type),
            description=med_info.get('description', f'{med_type} therapy')
        )
        scenario.set_medication(med_type, med_info.get('effects', {}))
        return scenario
    
    def create_combined_intervention(self) -> InterventionScenario:
        """Create combined lifestyle + medication scenario"""
        scenario = InterventionScenario(
            name="Combined Intervention",
            description="Lifestyle changes + medication"
        )
        scenario.set_lifestyle_change('physical_activity', 'vigorous')
        scenario.set_lifestyle_change('diet_quality', 'excellent')
        scenario.set_lifestyle_change('stress_level', 'low')
        scenario.set_weight_change(-10.0)
        scenario.set_medication('metformin', {'glucose_reduction': 20, 'hba1c_reduction': 0.5})
        return scenario
    
    def _apply_scenario_to_data(self, patient_data: Dict, scenario: InterventionScenario) -> Dict:
        """Apply scenario modifications to patient data"""
        modified_data = copy.deepcopy(patient_data)
        
        # Apply lifestyle changes
        if scenario.lifestyle_changes:
            if 'lifestyle' not in modified_data:
                modified_data['lifestyle'] = {}
            modified_data['lifestyle'].update(scenario.lifestyle_changes)
        
        # Apply weight change
        if scenario.weight_change != 0:
            current_weight = modified_data.get('weight', 70.0)
            new_weight = current_weight + scenario.weight_change
            modified_data['weight'] = max(40, new_weight)  # Minimum 40kg
            
            # Recalculate BMI
            height_m = modified_data.get('height', 170) / 100
            modified_data['bmi'] = new_weight / (height_m ** 2)
        
        # Apply medication effects
        if scenario.medication_effects:
            for med_type, effects in scenario.medication_effects.items():
                # Glucose reduction
                if 'glucose_reduction' in effects:
                    current_glucose = modified_data.get('fasting_glucose', 100)
                    modified_data['fasting_glucose'] = current_glucose - effects['glucose_reduction']
                
                # HbA1c reduction
                if 'hba1c_reduction' in effects:
                    current_hba1c = modified_data.get('hba1c', 5.5)
                    modified_data['hba1c'] = current_hba1c - effects['hba1c_reduction']
                
                # LDL reduction
                if 'ldl_reduction' in effects:
                    current_ldl = modified_data.get('ldl_cholesterol', 100)
                    modified_data['ldl_cholesterol'] = current_ldl - effects['ldl_reduction']
                
                # BP reduction
                if 'bp_reduction' in effects:
                    bp = modified_data.get('blood_pressure', {'systolic': 120, 'diastolic': 80})
                    bp['systolic'] -= effects['bp_reduction']
                    bp['diastolic'] -= effects['bp_reduction'] * 0.6
                    modified_data['blood_pressure'] = bp
        
        return modified_data
    
    def simulate_scenario(self, scenario: InterventionScenario) -> Dict:
        """
        Simulate a single scenario
        
        Args:
            scenario: Intervention scenario
            
        Returns:
            Simulation results
        """
        # Apply scenario to patient data
        modified_data = self._apply_scenario_to_data(self.baseline_data, scenario)
        
        # Run simulation
        simulator = DigitalTwinSimulator(modified_data)
        results = simulator.simulate(years=scenario.duration_years, timestep='month')
        
        # Add scenario info
        results['scenario_name'] = scenario.name
        results['scenario_description'] = scenario.description
        results['interventions_applied'] = {
            'lifestyle_changes': scenario.lifestyle_changes,
            'weight_change': scenario.weight_change,
            'medications': list(scenario.medication_effects.keys())
        }
        
        return results
    
    def simulate_all(self) -> Dict:
        """
        Simulate all scenarios
        
        Returns:
            Comparison results for all scenarios
        """
        print(f"\n🔬 Running scenario simulations...")
        print(f"   Baseline patient: {self.baseline_data.get('patient_id', 'UNKNOWN')}")
        print(f"   Scenarios: {len(self.scenarios)}")
        
        self.results = {}
        
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n   [{i}/{len(self.scenarios)}] Simulating: {scenario.name}")
            self.results[scenario.name] = self.simulate_scenario(scenario)
        
        print("\n   ✓ All scenarios complete!")
        
        return self.results
    
    def compare_outcomes(self) -> Dict:
        """
        Compare outcomes across all scenarios
        
        Returns:
            Comparison summary
        """
        if not self.results:
            return {}
        
        comparison = {
            'scenarios': list(self.results.keys()),
            'disease_risks': {},
            'parameter_changes': {},
            'best_scenario': None,
            'worst_scenario': None
        }
        
        # Compare disease risks
        for scenario_name, results in self.results.items():
            predictions = results.get('disease_predictions', [])
            
            # Get top risk
            if predictions:
                top_risk = max(predictions, key=lambda x: x.get('probability', 0))
                comparison['disease_risks'][scenario_name] = {
                    'disease': top_risk.get('disease', 'Unknown'),
                    'probability': top_risk.get('probability', 0),
                    'time_to_onset_years': top_risk.get('time_to_onset_years', 10)
                }
        
        # Find best and worst scenarios
        if comparison['disease_risks']:
            best = min(comparison['disease_risks'].items(), 
                      key=lambda x: x[1]['probability'])
            worst = max(comparison['disease_risks'].items(), 
                       key=lambda x: x[1]['probability'])
            
            comparison['best_scenario'] = {
                'name': best[0],
                'risk_reduction': worst[1]['probability'] - best[1]['probability']
            }
            comparison['worst_scenario'] = worst[0]
        
        return comparison
    
    def generate_recommendation(self) -> str:
        """
        Generate recommendation based on scenario comparison
        
        Returns:
            Recommendation text
        """
        comparison = self.compare_outcomes()
        
        if not comparison.get('best_scenario'):
            return "Insufficient data for recommendation"
        
        best = comparison['best_scenario']
        risk_reduction = best['risk_reduction'] * 100
        
        recommendation = f"""
SCENARIO ANALYSIS RECOMMENDATION

Best Intervention: {best['name']}
Risk Reduction: {risk_reduction:.1f}% lower disease probability

This scenario shows the most favorable health trajectory based on:
- Disease risk reduction
- Parameter improvement
- Long-term sustainability

Recommended Actions:
"""
        
        # Get interventions from best scenario
        best_results = self.results.get(best['name'], {})
        interventions = best_results.get('interventions_applied', {})
        
        if interventions.get('lifestyle_changes'):
            recommendation += "\n1. Lifestyle Modifications:"
            for factor, value in interventions['lifestyle_changes'].items():
                recommendation += f"\n   - {factor.replace('_', ' ').title()}: {value}"
        
        if interventions.get('weight_change'):
            recommendation += f"\n\n2. Weight Management:"
            recommendation += f"\n   - Target: {abs(interventions['weight_change']):.1f}kg weight loss"
        
        if interventions.get('medications'):
            recommendation += f"\n\n3. Pharmacotherapy:"
            for med in interventions['medications']:
                recommendation += f"\n   - Consider: {med.title()}"
        
        return recommendation


# Convenience function
def compare_interventions(patient_data: Dict, years: int = 5) -> Dict:
    """Quick comparison of standard intervention scenarios"""
    simulator = ScenarioSimulator(patient_data)
    
    # Add standard scenarios
    simulator.add_scenario(simulator.create_baseline_scenario())
    simulator.add_scenario(simulator.create_lifestyle_intervention())
    simulator.add_scenario(simulator.create_weight_loss_scenario(10.0))
    simulator.add_scenario(simulator.create_combined_intervention())
    
    # Set duration
    for scenario in simulator.scenarios:
        scenario.set_duration(years)
    
    # Run simulations
    simulator.simulate_all()
    
    return {
        'results': simulator.results,
        'comparison': simulator.compare_outcomes(),
        'recommendation': simulator.generate_recommendation()
    }
