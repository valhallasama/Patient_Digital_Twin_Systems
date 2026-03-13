"""
Simulation-Based Intervention Testing
Instead of using literature values, run actual simulations to see impact
Shows how lifestyle changes affect each organ agent and overall disease risk
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from copy import deepcopy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirofish_engine.parallel_digital_patient import ParallelDigitalPatient
from mirofish_engine.lifestyle_simulator import LifestyleSimulator, PatientLifestyleProfile


class SimulationBasedInterventionTester:
    """
    Test interventions by running actual simulations
    Compare baseline vs intervention scenarios
    """
    
    def __init__(self):
        self.baseline_results = None
        self.intervention_results = {}
    
    def run_baseline_simulation(
        self,
        patient_id: str,
        seed_info: Dict[str, Any],
        lifestyle_profile: PatientLifestyleProfile,
        days: int = 1825
    ) -> Dict[str, Any]:
        """
        Run baseline simulation with current lifestyle
        """
        print(f"\n🔬 Running BASELINE simulation ({days} days)...")
        
        # Create patient and simulator
        digital_patient = ParallelDigitalPatient(patient_id, seed_info)
        lifestyle_sim = LifestyleSimulator(lifestyle_profile)
        
        # Track organ changes
        organ_timeline = {
            'cardiovascular': [],
            'metabolic': [],
            'renal': [],
            'hepatic': [],
            'immune': [],
            'endocrine': [],
            'neural': []
        }
        
        timeline = []
        for day in range(days):
            daily_inputs = lifestyle_sim.get_daily_inputs(day)
            digital_patient.environment.external_inputs.update(daily_inputs)
            day_state = digital_patient._simulate_one_day()
            timeline.append(day_state)
            
            # Track each organ's state
            for organ_name in organ_timeline.keys():
                if organ_name in day_state['agents']:
                    organ_timeline[organ_name].append({
                        'day': day,
                        'state': deepcopy(day_state['agents'][organ_name]['state']),
                        'stress_level': day_state['agents'][organ_name].get('stress_level', 0)
                    })
            
            # Check for disease emergence
            emerged_diseases = digital_patient._detect_disease_emergence()
            for disease in emerged_diseases:
                if disease.name not in [d.name for d in digital_patient.diseases_emerged]:
                    digital_patient.diseases_emerged.append(disease)
            
            if day % 365 == 0 and day > 0:
                print(f"  ✓ Year {day//365} complete")
        
        digital_patient.timeline = timeline
        digital_patient.current_day = days
        
        # Calculate final metrics
        final_agents = {name: agent.to_dict() for name, agent in digital_patient.agents.items()}
        
        self.baseline_results = {
            'patient': digital_patient,
            'timeline': timeline,
            'organ_timeline': organ_timeline,
            'diseases_emerged': digital_patient.diseases_emerged,
            'final_agents': final_agents,
            'lifestyle_profile': lifestyle_profile
        }
        
        print(f"  ✓ Baseline complete: {len(digital_patient.diseases_emerged)} diseases emerged")
        
        return self.baseline_results
    
    def run_intervention_simulation(
        self,
        intervention_name: str,
        patient_id: str,
        seed_info: Dict[str, Any],
        modified_lifestyle_profile: PatientLifestyleProfile,
        days: int = 1825
    ) -> Dict[str, Any]:
        """
        Run simulation with modified lifestyle (intervention)
        """
        print(f"\n🧪 Running INTERVENTION simulation: {intervention_name} ({days} days)...")
        
        # Create patient and simulator with modified lifestyle
        digital_patient = ParallelDigitalPatient(patient_id, seed_info)
        lifestyle_sim = LifestyleSimulator(modified_lifestyle_profile)
        
        # Track organ changes
        organ_timeline = {
            'cardiovascular': [],
            'metabolic': [],
            'renal': [],
            'hepatic': [],
            'immune': [],
            'endocrine': [],
            'neural': []
        }
        
        timeline = []
        for day in range(days):
            daily_inputs = lifestyle_sim.get_daily_inputs(day)
            digital_patient.environment.external_inputs.update(daily_inputs)
            day_state = digital_patient._simulate_one_day()
            timeline.append(day_state)
            
            # Track each organ's state
            for organ_name in organ_timeline.keys():
                if organ_name in day_state['agents']:
                    organ_timeline[organ_name].append({
                        'day': day,
                        'state': deepcopy(day_state['agents'][organ_name]['state']),
                        'stress_level': day_state['agents'][organ_name].get('stress_level', 0)
                    })
            
            # Check for disease emergence
            emerged_diseases = digital_patient._detect_disease_emergence()
            for disease in emerged_diseases:
                if disease.name not in [d.name for d in digital_patient.diseases_emerged]:
                    digital_patient.diseases_emerged.append(disease)
            
            if day % 365 == 0 and day > 0:
                print(f"  ✓ Year {day//365} complete")
        
        digital_patient.timeline = timeline
        digital_patient.current_day = days
        
        # Calculate final metrics
        final_agents = {name: agent.to_dict() for name, agent in digital_patient.agents.items()}
        
        result = {
            'patient': digital_patient,
            'timeline': timeline,
            'organ_timeline': organ_timeline,
            'diseases_emerged': digital_patient.diseases_emerged,
            'final_agents': final_agents,
            'lifestyle_profile': modified_lifestyle_profile
        }
        
        self.intervention_results[intervention_name] = result
        
        print(f"  ✓ Intervention complete: {len(digital_patient.diseases_emerged)} diseases emerged")
        
        return result
    
    def compare_organ_changes(
        self,
        intervention_name: str,
        organ_name: str
    ) -> Dict[str, Any]:
        """
        Compare how a specific organ changed between baseline and intervention
        """
        if not self.baseline_results or intervention_name not in self.intervention_results:
            return {}
        
        baseline_organ = self.baseline_results['organ_timeline'][organ_name]
        intervention_organ = self.intervention_results[intervention_name]['organ_timeline'][organ_name]
        
        # Compare key metrics at different time points
        comparisons = []
        
        # Sample at yearly intervals
        for year in range(1, 6):
            day = year * 365 - 1
            if day < len(baseline_organ) and day < len(intervention_organ):
                baseline_state = baseline_organ[day]['state']
                intervention_state = intervention_organ[day]['state']
                
                comparison = {
                    'year': year,
                    'baseline': baseline_state,
                    'intervention': intervention_state,
                    'changes': {}
                }
                
                # Calculate changes for each metric
                for key in baseline_state.keys():
                    if isinstance(baseline_state[key], (int, float)):
                        baseline_val = baseline_state[key]
                        intervention_val = intervention_state[key]
                        change = intervention_val - baseline_val
                        pct_change = (change / baseline_val * 100) if baseline_val != 0 else 0
                        
                        comparison['changes'][key] = {
                            'baseline': baseline_val,
                            'intervention': intervention_val,
                            'absolute_change': change,
                            'percent_change': pct_change
                        }
                
                comparisons.append(comparison)
        
        return {
            'organ': organ_name,
            'intervention': intervention_name,
            'yearly_comparisons': comparisons
        }
    
    def calculate_intervention_impact(
        self,
        intervention_name: str
    ) -> Dict[str, Any]:
        """
        Calculate the actual impact of intervention from simulation results
        """
        if not self.baseline_results or intervention_name not in self.intervention_results:
            return {}
        
        baseline = self.baseline_results
        intervention = self.intervention_results[intervention_name]
        
        # Compare disease emergence
        baseline_diseases = {d.name: d.probability for d in baseline['diseases_emerged']}
        intervention_diseases = {d.name: d.probability for d in intervention['diseases_emerged']}
        
        # Calculate risk reduction
        disease_impacts = {}
        for disease_name in baseline_diseases.keys():
            baseline_risk = baseline_diseases.get(disease_name, 0)
            intervention_risk = intervention_diseases.get(disease_name, 0)
            
            absolute_reduction = baseline_risk - intervention_risk
            relative_reduction = (absolute_reduction / baseline_risk * 100) if baseline_risk > 0 else 0
            
            disease_impacts[disease_name] = {
                'baseline_risk': baseline_risk,
                'intervention_risk': intervention_risk,
                'absolute_reduction': absolute_reduction,
                'relative_reduction': relative_reduction
            }
        
        # Check for diseases that didn't emerge in intervention
        prevented_diseases = []
        for disease_name in baseline_diseases.keys():
            if disease_name not in intervention_diseases:
                prevented_diseases.append(disease_name)
        
        # Compare final organ states
        organ_comparisons = {}
        for organ_name in ['cardiovascular', 'metabolic', 'renal', 'hepatic']:
            organ_comp = self.compare_organ_changes(intervention_name, organ_name)
            if organ_comp:
                organ_comparisons[organ_name] = organ_comp
        
        return {
            'intervention': intervention_name,
            'disease_impacts': disease_impacts,
            'prevented_diseases': prevented_diseases,
            'organ_comparisons': organ_comparisons,
            'baseline_lifestyle': baseline['lifestyle_profile'],
            'intervention_lifestyle': intervention['lifestyle_profile']
        }
    
    def generate_simulation_based_recommendation(
        self,
        intervention_name: str
    ) -> str:
        """
        Generate recommendation based on actual simulation results
        NOT from literature, but from what we observed in the simulation
        """
        impact = self.calculate_intervention_impact(intervention_name)
        
        if not impact:
            return "No impact data available"
        
        report = f"\n{'='*80}\n"
        report += f"SIMULATION-BASED INTERVENTION ANALYSIS: {intervention_name}\n"
        report += f"{'='*80}\n\n"
        
        # Lifestyle changes
        baseline_ls = impact['baseline_lifestyle']
        intervention_ls = impact['intervention_lifestyle']
        
        report += "📊 Lifestyle Changes Applied:\n"
        report += f"  • Exercise: {baseline_ls.exercise_frequency} → {intervention_ls.exercise_frequency}\n"
        report += f"  • Diet: {baseline_ls.diet_quality} → {intervention_ls.diet_quality}\n"
        report += f"  • Sleep: {baseline_ls.sleep_pattern} → {intervention_ls.sleep_pattern}\n"
        report += f"  • Stress: {baseline_ls.stress_level} → {intervention_ls.stress_level}\n\n"
        
        # Disease impact
        report += "🎯 Disease Risk Impact (from simulation):\n"
        for disease, data in impact['disease_impacts'].items():
            report += f"\n  {disease}:\n"
            report += f"    Baseline risk: {data['baseline_risk']:.1%}\n"
            report += f"    With intervention: {data['intervention_risk']:.1%}\n"
            report += f"    Risk reduction: {data['absolute_reduction']:.1%} ({data['relative_reduction']:.1f}%)\n"
        
        if impact['prevented_diseases']:
            report += f"\n  ✅ Diseases PREVENTED: {', '.join(impact['prevented_diseases'])}\n"
        
        # Organ-level changes
        report += f"\n🔬 How Intervention Affected Each Organ:\n"
        
        for organ_name, organ_data in impact['organ_comparisons'].items():
            report += f"\n  {organ_name.upper()}:\n"
            
            # Show year 5 comparison
            if organ_data['yearly_comparisons']:
                year5 = organ_data['yearly_comparisons'][-1]
                
                # Show top 3 most changed metrics
                changes = year5['changes']
                sorted_changes = sorted(
                    changes.items(),
                    key=lambda x: abs(x[1]['percent_change']),
                    reverse=True
                )[:3]
                
                for metric, change_data in sorted_changes:
                    if abs(change_data['percent_change']) > 1:  # Only show significant changes
                        direction = "↓" if change_data['absolute_change'] < 0 else "↑"
                        report += f"    • {metric}: {change_data['baseline']:.2f} → {change_data['intervention']:.2f} "
                        report += f"({direction}{abs(change_data['percent_change']):.1f}%)\n"
        
        report += f"\n{'='*80}\n"
        report += "💡 This analysis is based on ACTUAL SIMULATION, not literature estimates!\n"
        report += f"{'='*80}\n"
        
        return report


def create_intervention_scenarios(
    base_lifestyle: PatientLifestyleProfile
) -> Dict[str, PatientLifestyleProfile]:
    """
    Create different intervention scenarios to test
    """
    scenarios = {}
    
    # Scenario 1: Increase exercise (30min/week → 150min/week)
    exercise_profile = PatientLifestyleProfile(
        occupation=base_lifestyle.occupation,
        exercise_frequency='high',  # 5 sessions/week
        diet_quality=base_lifestyle.diet_quality,
        sleep_pattern=base_lifestyle.sleep_pattern,
        stress_level=base_lifestyle.stress_level
    )
    scenarios['exercise_increase'] = exercise_profile
    
    # Scenario 2: Improve diet
    diet_profile = PatientLifestyleProfile(
        occupation=base_lifestyle.occupation,
        exercise_frequency=base_lifestyle.exercise_frequency,
        diet_quality='good',  # Mediterranean/healthy diet
        sleep_pattern=base_lifestyle.sleep_pattern,
        stress_level=base_lifestyle.stress_level
    )
    scenarios['diet_improvement'] = diet_profile
    
    # Scenario 3: Better sleep
    sleep_profile = PatientLifestyleProfile(
        occupation=base_lifestyle.occupation,
        exercise_frequency=base_lifestyle.exercise_frequency,
        diet_quality=base_lifestyle.diet_quality,
        sleep_pattern='good',  # 7-8h/night
        stress_level=base_lifestyle.stress_level
    )
    scenarios['sleep_improvement'] = sleep_profile
    
    # Scenario 4: Combined intervention (exercise + diet + sleep)
    combined_profile = PatientLifestyleProfile(
        occupation=base_lifestyle.occupation,
        exercise_frequency='high',
        diet_quality='good',
        sleep_pattern='good',
        stress_level='low'
    )
    scenarios['combined_intervention'] = combined_profile
    
    return scenarios


# Global instance
_tester = None

def get_simulation_tester() -> SimulationBasedInterventionTester:
    """Get or create global simulation tester"""
    global _tester
    if _tester is None:
        _tester = SimulationBasedInterventionTester()
    return _tester
