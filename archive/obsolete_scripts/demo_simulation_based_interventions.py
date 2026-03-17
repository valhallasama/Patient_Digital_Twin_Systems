#!/usr/bin/env python3
"""
SIMULATION-BASED INTERVENTION TESTING
Instead of using literature values, run actual simulations to see impact
Shows how lifestyle changes affect each organ agent and overall disease risk
Uses Qwen LLM to explain complex organ interactions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.simulation_based_interventions import (
    get_simulation_tester,
    create_intervention_scenarios
)
from utils.qwen_explainer import get_qwen_explainer
from mirofish_engine.lifestyle_simulator import PatientLifestyleProfile
import os


def main():
    print("\n" + "="*80)
    print("SIMULATION-BASED INTERVENTION TESTING")
    print("Real Simulations, Not Literature Estimates!")
    print("="*80)
    
    # Check for Qwen API key
    qwen_key = os.environ.get('QWEN_API_KEY')
    if qwen_key:
        print("\n✓ Qwen API key found - will use LLM for explanations")
    else:
        print("\n⚠️  No Qwen API key - using rule-based explanations")
        print("   To enable Qwen LLM: export QWEN_API_KEY='your-key'")
        print("   Get key from: https://bailian.console.aliyun.com/")
    
    # Patient seed information
    seed_info = {
        'patient_id': 'DT-SIM-0426',
        'initial_composition': {
            'glucose': 5.8,
            'cortisol': 1.3,
            'ldl': 3.6,
            'hdl': 1.1,
            'inflammation_markers': 0.18
        },
        'agent_seeds': {
            'cardiovascular': {
                'initial_state': {
                    'systolic_bp': 132,
                    'diastolic_bp': 86,
                    'heart_rate': 72,
                    'cardiac_output': 5.0,
                    'vessel_elasticity': 0.88,
                    'atherosclerosis_level': 0.12
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.4,
                'cooperation': 0.7
            },
            'metabolic': {
                'initial_state': {
                    'glucose': 5.8,
                    'insulin': 14.0,
                    'hba1c': 5.7,
                    'insulin_sensitivity': 0.70,
                    'beta_cell_function': 0.85,
                    'insulin_resistance': 0.30
                },
                'resilience': 0.4,
                'reactivity': 0.7,
                'adaptability': 0.3,
                'cooperation': 0.6
            },
            'renal': {
                'initial_state': {
                    'egfr': 95,
                    'creatinine': 92,
                    'filtration_capacity': 0.95,
                    'damage_level': 0.05
                },
                'resilience': 0.6,
                'reactivity': 0.5,
                'adaptability': 0.5,
                'cooperation': 0.7
            },
            'hepatic': {
                'initial_state': {
                    'alt': 42,
                    'ast': 30,
                    'ldl': 3.6,
                    'hdl': 1.1,
                    'fat_content': 0.08,
                    'detox_capacity': 0.92
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.4,
                'cooperation': 0.6
            },
            'immune': {
                'initial_state': {
                    'wbc': 6.8,
                    'crp': 1.8,
                    'inflammation': 0.22,
                    'immune_activation': 0.25
                },
                'resilience': 0.5,
                'reactivity': 0.8,
                'adaptability': 0.4,
                'cooperation': 0.5
            },
            'endocrine': {
                'initial_state': {
                    'cortisol': 1.3,
                    'thyroid': 1.0,
                    'stress_response': 0.4
                },
                'resilience': 0.4,
                'reactivity': 0.7,
                'adaptability': 0.4,
                'cooperation': 0.6
            },
            'neural': {
                'initial_state': {
                    'stress_level': 0.5,
                    'sleep_quality': 0.60,
                    'cognitive_function': 0.95
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.5,
                'cooperation': 0.7
            }
        }
    }
    
    # Baseline lifestyle (poor)
    baseline_lifestyle = PatientLifestyleProfile(
        occupation='office_worker',
        exercise_frequency='low',  # 1-2 sessions/week (~30 min/week)
        diet_quality='poor',
        sleep_pattern='insufficient',
        stress_level='moderate'
    )
    
    print("\n📋 Patient Profile:")
    print(f"  • Exercise: {baseline_lifestyle.exercise_frequency} (1-2 sessions/week, ~30 min total)")
    print(f"  • Diet: {baseline_lifestyle.diet_quality}")
    print(f"  • Sleep: {baseline_lifestyle.sleep_pattern} (6.5h/night)")
    print(f"  • Stress: {baseline_lifestyle.stress_level}")
    
    # Initialize tester
    tester = get_simulation_tester()
    
    # Run baseline simulation
    print("\n" + "="*80)
    print("STEP 1: Running BASELINE Simulation (Current Lifestyle)")
    print("="*80)
    
    baseline_results = tester.run_baseline_simulation(
        'DT-SIM-0426',
        seed_info,
        baseline_lifestyle,
        days=1825  # 5 years
    )
    
    print(f"\n📊 Baseline Results:")
    print(f"  • Diseases emerged: {len(baseline_results['diseases_emerged'])}")
    for disease in baseline_results['diseases_emerged']:
        print(f"    - {disease.name}: {disease.probability:.0%} at day {disease.day_emerged}")
    
    # Create intervention scenarios
    print("\n" + "="*80)
    print("STEP 2: Testing INTERVENTION Scenarios")
    print("="*80)
    
    scenarios = create_intervention_scenarios(baseline_lifestyle)
    
    # Test: Exercise increase (30 min/week → 150 min/week)
    print("\n🏃 Testing: EXERCISE INCREASE")
    print("  From: 1-2 sessions/week (~30 min)")
    print("  To: 5 sessions/week (~150 min)")
    
    exercise_results = tester.run_intervention_simulation(
        'exercise_increase',
        'DT-SIM-0426',
        seed_info,
        scenarios['exercise_increase'],
        days=1825
    )
    
    print(f"\n📊 Exercise Intervention Results:")
    print(f"  • Diseases emerged: {len(exercise_results['diseases_emerged'])}")
    for disease in exercise_results['diseases_emerged']:
        print(f"    - {disease.name}: {disease.probability:.0%} at day {disease.day_emerged}")
    
    # Calculate impact
    print("\n" + "="*80)
    print("STEP 3: Analyzing SIMULATION-BASED Impact")
    print("="*80)
    
    impact = tester.calculate_intervention_impact('exercise_increase')
    
    print("\n🎯 Disease Risk Changes (FROM SIMULATION):")
    for disease, data in impact['disease_impacts'].items():
        print(f"\n  {disease}:")
        print(f"    Baseline: {data['baseline_risk']:.1%}")
        print(f"    With exercise: {data['intervention_risk']:.1%}")
        print(f"    Reduction: {data['absolute_reduction']:.1%} ({data['relative_reduction']:.1f}%)")
    
    if impact['prevented_diseases']:
        print(f"\n  ✅ Diseases PREVENTED: {', '.join(impact['prevented_diseases'])}")
    
    # Explain organ-level changes
    print("\n" + "="*80)
    print("STEP 4: How Exercise Affected Each Organ (Simulation Results)")
    print("="*80)
    
    explainer = get_qwen_explainer()
    
    for organ_name, organ_data in impact['organ_comparisons'].items():
        print(f"\n🔬 {organ_name.upper()}:")
        
        if organ_data['yearly_comparisons']:
            # Get year 5 data
            year5 = organ_data['yearly_comparisons'][-1]
            baseline_state = year5['baseline']
            intervention_state = year5['intervention']
            
            # Show key changes
            print(f"\n  Key Changes (Year 5):")
            changes = year5['changes']
            sorted_changes = sorted(
                changes.items(),
                key=lambda x: abs(x[1]['percent_change']),
                reverse=True
            )[:3]
            
            for metric, change_data in sorted_changes:
                if abs(change_data['percent_change']) > 1:
                    direction = "↓" if change_data['absolute_change'] < 0 else "↑"
                    print(f"    • {metric}: {change_data['baseline']:.3f} → {change_data['intervention']:.3f} "
                          f"({direction}{abs(change_data['percent_change']):.1f}%)")
            
            # Get LLM/rule-based explanation
            explanation = explainer.explain_organ_changes(
                organ_name,
                baseline_state,
                intervention_state,
                "increased exercise from 30 to 150 min/week"
            )
            
            print(f"\n  💡 Explanation:")
            print(f"     {explanation}")
    
    # Explain cascade
    print("\n" + "="*80)
    print("STEP 5: Understanding the Cascade of Changes")
    print("="*80)
    
    # Prepare organ changes summary for cascade explanation
    organ_changes_summary = {}
    for organ_name, organ_data in impact['organ_comparisons'].items():
        if organ_data['yearly_comparisons']:
            year5 = organ_data['yearly_comparisons'][-1]
            organ_changes_summary[organ_name] = {
                'significant_changes': []
            }
            
            for metric, change_data in year5['changes'].items():
                if abs(change_data['percent_change']) > 5:
                    organ_changes_summary[organ_name]['significant_changes'].append({
                        'metric': metric,
                        'percent_change': change_data['percent_change']
                    })
    
    cascade_explanation = explainer.explain_intervention_cascade(
        "increased exercise from 30 to 150 min/week",
        organ_changes_summary
    )
    
    print(f"\n💡 How the Body Systems Responded Together:\n")
    print(f"{cascade_explanation}\n")
    
    # Generate final report
    report = tester.generate_simulation_based_recommendation('exercise_increase')
    print(report)
    
    # Summary
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("""
✅ What This Demonstrates:

1. 🔬 SIMULATION-BASED Analysis
   - NOT using literature estimates (e.g., "NEJM says 58% reduction")
   - ACTUAL simulation showing how organs respond
   - Real cause-and-effect from lifestyle → organs → disease

2. 🎯 Organ-Level Understanding
   - See exactly which organs improved
   - Understand WHY they improved (mechanisms)
   - Track the cascade of changes across systems

3. 💊 Personalized Recommendations
   - Based on YOUR patient's simulation
   - Shows actual impact in YOUR patient's body
   - Not generic population statistics

4. 🤖 LLM-Enhanced Explanations (if Qwen enabled)
   - Complex organ interactions explained clearly
   - Physiological mechanisms described
   - Causal chains from intervention to outcome

This is TRUE personalized medicine - predictions based on simulating
YOUR patient's body, not population averages! 🚀
""")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
